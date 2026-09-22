"""Food Demand Forecasting API  (SIC AI 17 - Grup 11)

Servis edilen modeller (LightGBM, log1p hedef):
  point  -> nokta tahmini            (final_submission_lgb_model.txt)
  q_low  -> alt band  alpha=0.08     (quantile_final_p8.txt)
  q_mid  -> medyan    alpha=0.50     (quantile_final_p50.txt)
  q_high -> ust band  alpha=0.92     (quantile_final_p92.txt)
Ozellik uretimi extensions/forecasting/pipeline.py ile birebir aynidir.
"""
from __future__ import annotations

import hashlib, json, logging, math, os, secrets, time, uuid
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock
from typing import List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parents[1] / "models"))
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", Path(__file__).resolve().parents[2] / "frontend"))
PRED_LOG = Path(os.getenv("PREDICTION_LOG", "logs/predictions.jsonl"))
API_KEYS = {k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()}
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "120"))
ALLOWED_ORIGINS = [o for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o]
MAX_BATCH = 500
API_VERSION = "1.0.0"

# ------------------------------------------------------------------ logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("food-demand-api")

def jlog(event: str, **fields) -> None:
    log.info(json.dumps({"ts": time.time(), "event": event, **fields}, default=str))

# ------------------------------------------------------------------ metrics
REQUESTS = Counter("api_requests_total", "HTTP istekleri", ["path", "method", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Istek suresi", ["path"],
                    buckets=(.01, .025, .05, .1, .25, .5, 1, 2.5, 5))
PRED_VALUE = Histogram("prediction_num_orders", "Tahmin edilen siparis (drift izleme)",
                       buckets=(25, 50, 100, 200, 300, 500, 1000, 2000, 5000))
COLD_START = Counter("prediction_cold_start_total", "Gecmisi olmayan (lag=NaN) tahminler")
AUTH_FAIL = Counter("auth_failures_total", "Gecersiz API key denemeleri")
RATE_LIMITED = Counter("rate_limited_total", "Rate limit'e takilan istekler")

# ------------------------------------------------------------------ model yukleme
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def load_artifacts(model_dir: Path | None = None):
    model_dir = (model_dir or MODEL_DIR).resolve()
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    maps = json.loads((model_dir / "category_maps.json").read_text(encoding="utf-8"))
    boosters = {}
    for key, meta in manifest["models"].items():
        path = (model_dir / meta["file"]).resolve()
        if not path.is_relative_to(model_dir) or not path.is_file():
            raise RuntimeError(f"Invalid model artifact path: {meta['file']}")
        if _sha256(path) != meta["sha256"]:          # butunluk kontrolu
            raise RuntimeError(f"Model hash uyusmuyor: {path.name}")
        boosters[key] = lgb.Booster(model_file=str(path))
    return manifest, maps, boosters

MANIFEST, CATEGORY_MAPS, BOOSTERS = load_artifacts()
FEATURES: List[str] = MANIFEST["feature_columns"]
jlog("models_loaded", models=list(BOOSTERS), features=len(FEATURES))

# ------------------------------------------------------------------ sema
class DemandRequest(BaseModel):
    week: int = Field(ge=1, le=520, description="Hafta numarasi (1..)")
    center_id: int = Field(ge=0)
    meal_id: int = Field(ge=0)
    checkout_price: float = Field(gt=0, le=10_000)
    base_price: float = Field(gt=0, le=10_000)
    emailer_for_promotion: int = Field(ge=0, le=1)
    homepage_featured: int = Field(ge=0, le=1)
    category: str
    cuisine: str
    center_type: str
    city_code: int = Field(ge=0)
    region_code: int = Field(ge=0)
    op_area: float = Field(gt=0, le=100)
    recent_orders: List[float] = Field(
        default_factory=list, max_length=13,
        description="Bu center+meal icin gecmis haftalarin GERCEK siparisleri (eskiden yeniye). "
                    "En az 4 hafta verilirse lag_4 ve roll_mean_4 hesaplanir.")

class DemandResponse(BaseModel):
    predicted_orders: float
    p_low: float
    p_median: float
    p_high: float
    interval_nominal: str = "alpha 0.08-0.92 (validation coverage ~%80.6)"
    model_version: str
    warnings: List[str] = Field(default_factory=list)

class BatchResponse(BaseModel):
    count: int
    results: List[DemandResponse]

# ------------------------------------------------------------------ ozellik uretimi
def build_features(r: DemandRequest) -> tuple[dict, List[str]]:
    warnings: List[str] = []
    for col, val in (("center_type", r.center_type), ("category", r.category), ("cuisine", r.cuisine)):
        if val not in CATEGORY_MAPS[col]:
            raise HTTPException(422, f"Bilinmeyen {col}: '{val}'. Gecerli: {sorted(CATEGORY_MAPS[col])}")
    if any(v < 0 or not math.isfinite(v) for v in r.recent_orders):
        raise HTTPException(422, "recent_orders negatif veya sonsuz deger iceremez")
    h = r.recent_orders
    woy = ((r.week - 1) % 52) + 1
    row = {
        "checkout_price": r.checkout_price,
        "base_price": r.base_price,
        "discount_ratio": (r.base_price - r.checkout_price) / r.base_price,
        "emailer_for_promotion": r.emailer_for_promotion,
        "homepage_featured": r.homepage_featured,
        "num_orders_lag_1": h[-1] if len(h) >= 1 else np.nan,
        "num_orders_lag_4": h[-4] if len(h) >= 4 else np.nan,
        "num_orders_roll_mean_4": float(np.mean(h[-4:])) if len(h) >= 4 else np.nan,
        "weekofyear": woy,
        "week_sin": math.sin(2 * math.pi * woy / 52),
        "week_cos": math.cos(2 * math.pi * woy / 52),
        "center_type_enc": CATEGORY_MAPS["center_type"][r.center_type],
        "category_enc": CATEGORY_MAPS["category"][r.category],
        "cuisine_enc": CATEGORY_MAPS["cuisine"][r.cuisine],
        "city_code": r.city_code, "region_code": r.region_code, "op_area": r.op_area,
        "center_id": r.center_id, "meal_id": r.meal_id,
    }
    if len(h) < 4:
        warnings.append("cold_start: 4 haftadan az gecmis verildi, lag ozellikleri eksik (NaN); "
                        "tahmin belirsizligi yuksek")
        COLD_START.inc()
    return row, warnings

def predict_rows(rows: List[dict], warns: List[List[str]]) -> List[DemandResponse]:
    X = pd.DataFrame(rows)[FEATURES]
    out = {k: np.clip(np.expm1(b.predict(X)), 0, None) for k, b in BOOSTERS.items()}
    band = np.sort(np.vstack([out["q_low"], out["q_mid"], out["q_high"]]), axis=0)  # quantile crossing duzelt
    results = []
    for i in range(len(rows)):
        PRED_VALUE.observe(float(out["point"][i]))
        results.append(DemandResponse(
            predicted_orders=round(float(out["point"][i]), 2),
            p_low=round(float(band[0, i]), 2), p_median=round(float(band[1, i]), 2),
            p_high=round(float(band[2, i]), 2),
            model_version=f"{API_VERSION}+{MANIFEST['models']['point']['sha256'][:8]}",
            warnings=warns[i]))
    return results

_prediction_log_lock = Lock()


def log_predictions(reqs: List[DemandRequest], res: List[DemandResponse], rid: str) -> None:
    """Tahminleri kaydet -> sonradan gercek satislarla karsilastirmak icin (model performans izleme)."""
    try:
        PRED_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _prediction_log_lock, open(PRED_LOG, "a", encoding="utf-8") as f:
            for q, p in zip(reqs, res, strict=True):
                f.write(json.dumps({"request_id": rid, "ts": time.time(), "week": q.week,
                                    "center_id": q.center_id, "meal_id": q.meal_id,
                                    "n_history": len(q.recent_orders),
                                    "features": q.model_dump(exclude={"recent_orders"}),
                                    "pred": p.predicted_orders, "p_low": p.p_low,
                                    "p_high": p.p_high, "model_version": p.model_version}) + "\n")
    except OSError as e:  # loglama hatasi tahmini engellememeli
        jlog("prediction_log_error", error=str(e))

# ------------------------------------------------------------------ guvenlik
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_hits: dict[str, deque] = defaultdict(deque)
_rate_limit_lock = Lock()

def verify_key(key: Optional[str] = Security(api_key_header)) -> str:
    if not API_KEYS:
        raise HTTPException(503, "Sunucuda API_KEYS tanimli degil")
    if not key or not any(secrets.compare_digest(key, k) for k in API_KEYS):  # sabit-zamanli karsilastirma
        AUTH_FAIL.inc()
        raise HTTPException(401, "Gecersiz veya eksik API key")
    now = time.time()
    with _rate_limit_lock:
        q = _hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT_PER_MIN:
            RATE_LIMITED.inc()
            raise HTTPException(429, "Rate limit asildi", headers={"Retry-After": "60"})
        q.append(now)
    return key

# ------------------------------------------------------------------ uygulama
app = FastAPI(title="Food Demand Forecasting API", version=API_VERSION,
              docs_url="/docs", redoc_url=None)
if ALLOWED_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
                       allow_methods=["GET", "POST"], allow_headers=["X-API-Key", "Content-Type"])

@app.middleware("http")
async def observe(request: Request, call_next):
    rid, start = str(uuid.uuid4()), time.perf_counter()
    request.state.rid = rid
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response
    except Exception:
        log.exception(json.dumps({"event": "unhandled_error", "request_id": rid}))
        return Response("Internal Server Error", status_code=500)
    finally:
        dt = time.perf_counter() - start
        REQUESTS.labels(request.url.path, request.method, status).inc()
        LATENCY.labels(request.url.path).observe(dt)
        jlog("request", request_id=rid, path=request.url.path, method=request.method,
             status=status, latency_ms=round(dt * 1000, 1))

@app.get("/health")
def health():
    return {"status": "ok", "version": API_VERSION, "models": list(BOOSTERS)}

@app.get("/model-info", dependencies=[Depends(verify_key)])
def model_info():
    return {"feature_columns": FEATURES, "target_transform": MANIFEST["target_transform"],
            "training_weeks": MANIFEST["training_weeks"], "created_at_utc": MANIFEST["created_at_utc"],
            "categories": {k: sorted(v) for k, v in CATEGORY_MAPS.items()},
            "reported_metrics": {"holdout_rmsle": 0.47534, "walk_forward_rmsle": 0.49833,
                                 "persistence_rmsle": 0.80930, "band_coverage": 0.8059}}

@app.post("/predict", response_model=DemandResponse, dependencies=[Depends(verify_key)])
def predict(body: DemandRequest, request: Request):
    row, warns = build_features(body)
    res = predict_rows([row], [warns])
    log_predictions([body], res, request.state.rid)
    return res[0]

@app.post("/predict/batch", response_model=BatchResponse, dependencies=[Depends(verify_key)])
def predict_batch(body: List[DemandRequest], request: Request):
    if not body:
        raise HTTPException(422, "Bos liste")
    if len(body) > MAX_BATCH:
        raise HTTPException(413, f"Batch en fazla {MAX_BATCH} satir olabilir")
    built = [build_features(b) for b in body]
    res = predict_rows([b[0] for b in built], [b[1] for b in built])
    log_predictions(body, res, request.state.rid)
    return BatchResponse(count=len(res), results=res)

@app.get("/metrics", dependencies=[Depends(verify_key)])
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Frontend mount'u API rotalarindan sonra eklenir; boylece /docs, /health ve
# tahmin endpointleri statik dosya servisi tarafindan golgelenmez.
if FRONTEND_DIR.is_dir():
    @app.get("/", include_in_schema=False)
    def frontend_index():
        return FileResponse(
            FRONTEND_DIR / "index.html",
            headers={"Cache-Control": "no-store"},
        )

    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
