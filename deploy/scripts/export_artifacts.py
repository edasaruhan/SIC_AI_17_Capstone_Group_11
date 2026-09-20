"""Prepare trusted LightGBM artifacts for the deployment service.

The manifest is generated from the bytes that will actually be served. This
avoids cross-platform line-ending differences invalidating model hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "point": "final_submission_lgb_model.txt",
    "q_low": "quantile_final_p8.txt",
    "q_mid": "quantile_final_p50.txt",
    "q_high": "quantile_final_p92.txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export deployment model artifacts")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "deploy" / "models")
    parser.add_argument(
        "--spec",
        type=Path,
        default=ROOT / "deploy" / "models" / "model_spec.json",
    )
    parser.add_argument(
        "--category-maps",
        type=Path,
        default=ROOT / "deploy" / "models" / "category_maps.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_artifacts(
    source_dir: Path,
    output_dir: Path,
    spec_path: Path,
    category_maps_path: Path,
) -> dict[str, object]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    category_maps = json.loads(category_maps_path.read_text(encoding="utf-8"))
    if not spec.get("feature_columns"):
        raise ValueError("model_spec.json must contain feature_columns")
    if set(category_maps) != {"center_type", "category", "cuisine"}:
        raise ValueError("category_maps.json has an unexpected schema")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **spec,
        "models": {},
    }
    for key, name in MODELS.items():
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing model artifact: {source}")
        destination = output_dir / name
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        manifest["models"][key] = {
            "file": name,
            "sha256": sha256(destination),
            "size_mb": round(destination.stat().st_size / 1e6, 2),
        }

    (output_dir / "category_maps.json").write_text(
        json.dumps(category_maps, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = export_artifacts(
        args.source_dir,
        args.output_dir,
        args.spec,
        args.category_maps,
    )
    print(
        f"Prepared {len(manifest['models'])} models in {args.output_dir.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
