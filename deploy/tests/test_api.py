"""Integration tests for the FastAPI deployment service."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


os.environ["API_KEYS"] = "test-key"

from app import main as api  # noqa: E402


client = TestClient(api.app)
HEADERS = {"X-API-Key": "test-key"}
PAYLOAD = {
    "week": 136,
    "center_id": 10,
    "meal_id": 1062,
    "checkout_price": 194.06,
    "base_price": 194.06,
    "emailer_for_promotion": 0,
    "homepage_featured": 0,
    "category": "Beverages",
    "cuisine": "Italian",
    "center_type": "TYPE_B",
    "city_code": 590,
    "region_code": 56,
    "op_area": 6.3,
    "recent_orders": [1538, 700, 704, 960],
}


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(api, "PRED_LOG", tmp_path / "predictions.jsonl")
    monkeypatch.setattr(api, "RATE_LIMIT_PER_MIN", 120)
    api._hits.clear()
    yield
    api._hits.clear()


def test_health_open():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_frontend_shell_and_assets_are_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "Yemek Üretim Planlama" in response.text
    assert client.get("/frontend/styles.css").status_code == 200
    script = client.get("/frontend/app.js")
    assert script.status_code == 200
    assert "local-demo-key" not in script.text  # API anahtari kaynak koda gomulmez


def test_auth_required():
    assert client.post("/predict", json=PAYLOAD).status_code == 401
    assert client.post("/predict", json=PAYLOAD, headers={"X-API-Key": "wrong"}).status_code == 401


def test_features_match_training_contract():
    row, warnings = api.build_features(api.DemandRequest(**PAYLOAD))
    assert list(row) == api.FEATURES
    assert row["num_orders_lag_1"] == 960
    assert row["num_orders_lag_4"] == 1538
    assert math.isclose(row["num_orders_roll_mean_4"], 975.5)
    assert row["weekofyear"] == 32
    assert math.isclose(row["week_sin"], -0.663123, abs_tol=1e-5)
    assert (row["center_type_enc"], row["category_enc"], row["cuisine_enc"]) == (1, 0, 2)
    assert warnings == []


def test_predict_ok_band_ordered_and_logged():
    response = client.post("/predict", json=PAYLOAD, headers=HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["p_low"] <= body["p_median"] <= body["p_high"]
    assert 300 < body["predicted_orders"] < 2500
    lines = api.PRED_LOG.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    logged = json.loads(lines[0])
    assert logged["center_id"] == PAYLOAD["center_id"]
    assert logged["model_version"] == body["model_version"]


def test_cold_start_warning():
    response = client.post(
        "/predict",
        json={**PAYLOAD, "recent_orders": []},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["warnings"]


@pytest.mark.parametrize(
    "change",
    [
        {"checkout_price": -5},
        {"category": "Pizzaa"},
        {"recent_orders": [-1, 2, 3, 4]},
        {"emailer_for_promotion": 3},
    ],
)
def test_validation_errors(change):
    response = client.post("/predict", json={**PAYLOAD, **change}, headers=HEADERS)
    assert response.status_code == 422


def test_batch_and_limits():
    response = client.post(
        "/predict/batch",
        json=[PAYLOAD, {**PAYLOAD, "week": 137}],
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["count"] == 2
    assert client.post("/predict/batch", json=[], headers=HEADERS).status_code == 422
    assert client.post("/predict/batch", json=[PAYLOAD] * 501, headers=HEADERS).status_code == 413


def test_rate_limit_is_enforced(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(api, "RATE_LIMIT_PER_MIN", 1)
    assert client.get("/model-info", headers=HEADERS).status_code == 200
    response = client.get("/model-info", headers=HEADERS)
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "60"


def test_metrics_protected_and_populated():
    assert client.get("/metrics").status_code == 401
    response = client.get("/metrics", headers=HEADERS)
    assert response.status_code == 200
    assert "api_requests_total" in response.text
    assert "prediction_num_orders" in response.text


def test_model_roundtrip_preserves_prediction(tmp_path: Path):
    row, _ = api.build_features(api.DemandRequest(**PAYLOAD))
    features = pd.DataFrame([row])[api.FEATURES]
    expected = np.asarray(api.BOOSTERS["point"].predict(features))
    model_path = tmp_path / "point_model.txt"
    api.BOOSTERS["point"].save_model(str(model_path))
    restored = lgb.Booster(model_file=str(model_path))
    np.testing.assert_allclose(restored.predict(features), expected, rtol=0, atol=1e-12)


def test_tampered_model_is_rejected(tmp_path: Path):
    (tmp_path / "category_maps.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tampered.txt").write_text("not a model", encoding="utf-8")
    manifest = {
        "feature_columns": ["x"],
        "models": {"point": {"file": "tampered.txt", "sha256": "0" * 64}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash"):
        api.load_artifacts(tmp_path)
