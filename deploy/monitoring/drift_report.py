"""Create a PSI-based input and prediction drift report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "checkout_price",
    "base_price",
    "emailer_for_promotion",
    "homepage_featured",
]


def psi(expected, actual, bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if not len(expected) or not len(actual):
        raise ValueError("PSI requires non-empty finite samples")

    edges = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
    if len(edges) < 2:
        center = float(edges[0])
        margin = max(abs(center) * 0.01, 1e-6)
        edges = np.array([-np.inf, center - margin, center + margin, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    expected_share = np.histogram(expected, edges)[0] / len(expected) + 1e-6
    actual_share = np.histogram(actual, edges)[0] / len(actual) + 1e-6
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def normalize_prediction_log(rows: list[dict]) -> pd.DataFrame:
    current = pd.json_normalize(rows)
    renamed = {
        column: column.removeprefix("features.")
        for column in current.columns
        if column.startswith("features.")
    }
    return current.rename(columns=renamed)


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {"n_current": len(current), "features": {}}
    for column in FEATURES:
        result["features"][column] = psi(reference[column], current[column])
    reference_discount = (
        reference["base_price"] - reference["checkout_price"]
    ) / reference["base_price"]
    current_discount = (
        current["base_price"] - current["checkout_price"]
    ) / current["base_price"]
    result["features"]["discount_ratio"] = psi(reference_discount, current_discount)
    result["prediction"] = (
        psi(reference["pred_p50"], current["pred"])
        if "pred_p50" in reference and "pred" in current
        else None
    )
    result["cold_start_share"] = float((current["n_history"] < 4).mean())
    values = list(result["features"].values())
    if result["prediction"] is not None:
        values.append(result["prediction"])
    worst = max(values)
    result["status"] = "ALERT" if worst > 0.25 else "WARN" if worst > 0.10 else "OK"
    return result


def write_report(result: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--min-rows", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("logs/drift_report.json"))
    args = parser.parse_args()

    reference_path = Path(args.reference)
    reference = (
        pd.read_parquet(reference_path)
        if reference_path.suffix == ".parquet"
        else pd.read_csv(reference_path)
    )
    log_path = Path(args.log)
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) < args.min_rows:
        result = {
            "status": "INSUFFICIENT_DATA",
            "n_current": len(rows),
            "min_rows": args.min_rows,
        }
        write_report(result, args.out)
        print(json.dumps(result, indent=2))
        return 0

    result = compute_drift(reference, normalize_prediction_log(rows))
    write_report(result, args.out)
    print(json.dumps(result, indent=2))
    return 1 if result["status"] == "ALERT" else 0


if __name__ == "__main__":
    sys.exit(main())
