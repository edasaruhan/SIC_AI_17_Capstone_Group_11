"""Compare prediction logs with realized orders and emit performance alerts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE_RMSLE = 0.49833
RMSLE_ALERT = BASELINE_RMSLE * 1.10
COVERAGE_RANGE = (0.70, 0.90)
KEY_COLUMNS = ["week", "center_id", "meal_id"]


def rmsle(actual: pd.Series, predicted: pd.Series) -> float:
    return float(
        np.sqrt(
            np.mean(
                (np.log1p(actual.to_numpy(float)) - np.log1p(predicted.to_numpy(float))) ** 2
            )
        )
    )


def compute_performance(predictions: pd.DataFrame, actuals: pd.DataFrame) -> dict[str, object]:
    predictions = predictions.drop_duplicates(KEY_COLUMNS, keep="last")
    merged = predictions.merge(
        actuals[KEY_COLUMNS + ["num_orders"]],
        on=KEY_COLUMNS,
        how="inner",
    )
    if merged.empty:
        return {"status": "NO_MATCHING_DATA", "n_matched": 0, "alerts": []}

    errors = merged["pred"] - merged["num_orders"]
    absolute_errors = errors.abs()
    actual_total = float(merged["num_orders"].sum())
    result: dict[str, object] = {
        "n_matched": len(merged),
        "rmsle": rmsle(merged["num_orders"], merged["pred"]),
        "mae": float(absolute_errors.mean()),
        "wape": float(absolute_errors.sum() / actual_total) if actual_total > 0 else None,
        "bias": float(errors.mean()),
        "overforecast_rate": float((errors > 0).mean()),
        "underforecast_rate": float((errors < 0).mean()),
        "coverage": float(
            ((merged["num_orders"] >= merged["p_low"]) &
             (merged["num_orders"] <= merged["p_high"])).mean()
        ),
        "rmsle_by_history": {"cold_start(<4)": None, "warm(>=4)": None},
    }
    for name, mask in (
        ("cold_start(<4)", merged["n_history"] < 4),
        ("warm(>=4)", merged["n_history"] >= 4),
    ):
        subset = merged[mask]
        if len(subset):
            result["rmsle_by_history"][name] = rmsle(subset["num_orders"], subset["pred"])

    alerts: list[str] = []
    if result["rmsle"] > RMSLE_ALERT:
        alerts.append(f"RMSLE {result['rmsle']:.3f} > threshold {RMSLE_ALERT:.3f}")
    if not COVERAGE_RANGE[0] <= result["coverage"] <= COVERAGE_RANGE[1]:
        alerts.append(f"Band coverage {result['coverage']:.2%} is outside expected range")
    result["alerts"] = alerts
    result["status"] = "ALERT" if alerts else "OK"
    return result


def write_report(result: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--actuals", required=True)
    parser.add_argument("--out", type=Path, default=Path("logs/performance_report.json"))
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.log).read_text(encoding="utf-8").splitlines()
        if line
    ]
    predictions = pd.DataFrame(rows)
    actuals = pd.read_csv(args.actuals)
    result = compute_performance(predictions, actuals)
    write_report(result, args.out)
    print(json.dumps(result, indent=2))
    return 1 if result["status"] == "ALERT" else 0


if __name__ == "__main__":
    sys.exit(main())
