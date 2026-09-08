"""Calibrate a guarded base/long-memory ensemble on earlier folds and test it once."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pipeline import (  # noqa: E402
    FEATURE_COLUMNS,
    LONG_MEMORY_FEATURE_COLUMNS,
    PreparedData,
    prepare_data,
    recursive_predict,
    regression_metrics,
    train_point_model,
)


CALIBRATION_CUTOFFS = [115, 125]
FINAL_TEST_CUTOFF = 135
THRESHOLDS = [50.0, 75.0, 100.0, 150.0, 200.0, 300.0]
LONG_MEMORY_WEIGHTS = [0.25, 0.50, 0.75, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a guarded two-model forecast ensemble.")
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(__file__).parent / "outputs"
    parser.add_argument("--data-dir", type=Path, default=repo_root / "data")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--num-boost-round", type=int, default=1360)
    parser.add_argument("--output", type=Path, default=output_dir / "osman_hybrid_experiment.json")
    parser.add_argument("--figure", type=Path, default=output_dir / "osman_hybrid_experiment.png")
    return parser.parse_args()


def hybrid_predictions(
    base: np.ndarray,
    long_memory: np.ndarray,
    threshold: float,
    long_memory_weight: float,
) -> np.ndarray:
    base = np.asarray(base, dtype=float)
    long_memory = np.asarray(long_memory, dtype=float)
    if base.shape != long_memory.shape:
        raise ValueError("Base and long-memory predictions must have the same shape")
    if threshold < 0 or not 0 <= long_memory_weight <= 1:
        raise ValueError("Invalid hybrid threshold or weight")
    blended = (1 - long_memory_weight) * base + long_memory_weight * long_memory
    return np.where(base <= threshold, base, blended)


def score_predictions(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    metrics = regression_metrics(actual, predicted)
    residual = predicted - actual
    return {
        **metrics,
        "mae": float(np.mean(np.abs(residual))),
        "mean_bias": float(np.mean(residual)),
        "underforecast_rate": float(np.mean(predicted < actual)),
    }


def build_fold_predictions(
    prepared: PreparedData,
    cutoff: int,
    horizon: int,
    num_boost_round: int,
) -> pd.DataFrame:
    validation = prepared.train_static[
        (prepared.train_static["week"] > cutoff)
        & (prepared.train_static["week"] <= cutoff + horizon)
    ].copy()
    observed = prepared.train_static[prepared.train_static["week"] <= cutoff].copy()
    train_frame = prepared.train_observed[prepared.train_observed["week"] <= cutoff].copy()
    print(f"Fold {cutoff}: training base model", flush=True)
    base_model = train_point_model(
        train_frame,
        num_boost_round=num_boost_round,
        feature_columns=FEATURE_COLUMNS,
    )
    base = recursive_predict(
        base_model,
        validation,
        observed,
        feature_columns=FEATURE_COLUMNS,
    )
    print(f"Fold {cutoff}: training long-memory model", flush=True)
    long_model = train_point_model(
        train_frame,
        num_boost_round=num_boost_round,
        feature_columns=LONG_MEMORY_FEATURE_COLUMNS,
    )
    long_memory = recursive_predict(
        long_model,
        validation,
        observed,
        feature_columns=LONG_MEMORY_FEATURE_COLUMNS,
    )
    scored = validation[["id", "week", "num_orders"]].rename(
        columns={"num_orders": "actual"}
    )
    scored = scored.merge(
        base[["id", "num_orders"]].rename(columns={"num_orders": "base"}),
        on="id",
        validate="one_to_one",
    )
    scored = scored.merge(
        long_memory[["id", "num_orders"]].rename(columns={"num_orders": "long_memory"}),
        on="id",
        validate="one_to_one",
    )
    scored["fold_cutoff"] = cutoff
    return scored


def calibrate_hybrid(calibration: pd.DataFrame) -> dict[str, object]:
    actual = calibration["actual"].to_numpy(dtype=float)
    base = calibration["base"].to_numpy(dtype=float)
    long_memory = calibration["long_memory"].to_numpy(dtype=float)
    low_mask = actual <= 50
    base_overall = regression_metrics(actual, base)
    base_low = regression_metrics(actual[low_mask], base[low_mask])
    candidates: list[dict[str, object]] = []

    for threshold in THRESHOLDS:
        for weight in LONG_MEMORY_WEIGHTS:
            predicted = hybrid_predictions(base, long_memory, threshold, weight)
            overall = regression_metrics(actual, predicted)
            low = regression_metrics(actual[low_mask], predicted[low_mask])
            overall_improvement = (
                (base_overall["rmsle"] - overall["rmsle"]) / base_overall["rmsle"] * 100
            )
            low_change = (base_low["rmsle"] - low["rmsle"]) / base_low["rmsle"] * 100
            candidates.append(
                {
                    "threshold": threshold,
                    "long_memory_weight": weight,
                    "rmsle": overall["rmsle"],
                    "rmsle_improvement_pct": overall_improvement,
                    "low_demand_rmsle": low["rmsle"],
                    "low_demand_improvement_pct": low_change,
                    "passes_low_demand_guardrail": low_change >= -0.5,
                }
            )

    eligible = [row for row in candidates if row["passes_low_demand_guardrail"]]
    if not eligible:
        raise RuntimeError("No hybrid configuration passed the calibration guardrail")
    selected = min(eligible, key=lambda row: float(row["rmsle"]))
    return {
        "base_overall": base_overall,
        "base_low_demand": base_low,
        "selected": selected,
        "candidates_tested": len(candidates),
    }


def evaluate_final(scored: pd.DataFrame, selected: dict[str, object]) -> dict[str, object]:
    actual = scored["actual"].to_numpy(dtype=float)
    base = scored["base"].to_numpy(dtype=float)
    long_memory = scored["long_memory"].to_numpy(dtype=float)
    hybrid = hybrid_predictions(
        base,
        long_memory,
        float(selected["threshold"]),
        float(selected["long_memory_weight"]),
    )
    base_metrics = score_predictions(actual, base)
    long_metrics = score_predictions(actual, long_memory)
    hybrid_metrics = score_predictions(actual, hybrid)
    low_mask = actual <= 50
    high_mask = actual > 600
    base_low = regression_metrics(actual[low_mask], base[low_mask])
    hybrid_low = regression_metrics(actual[low_mask], hybrid[low_mask])
    base_high = regression_metrics(actual[high_mask], base[high_mask])
    hybrid_high = regression_metrics(actual[high_mask], hybrid[high_mask])
    overall_improvement = (
        (base_metrics["rmsle"] - hybrid_metrics["rmsle"]) / base_metrics["rmsle"] * 100
    )
    low_improvement = (
        (base_low["rmsle"] - hybrid_low["rmsle"]) / base_low["rmsle"] * 100
    )
    gates = {
        "final_rmsle_improves_at_least_0_2_pct": overall_improvement >= 0.2,
        "final_low_demand_regression_within_0_5_pct": low_improvement >= -0.5,
        "final_mae_not_worse": hybrid_metrics["mae"] <= base_metrics["mae"],
    }
    return {
        "decision": "PROMOTE" if all(gates.values()) else "REJECT",
        "gates": gates,
        "rows": len(scored),
        "base": base_metrics,
        "long_memory": long_metrics,
        "hybrid": hybrid_metrics,
        "rmsle_improvement_pct": overall_improvement,
        "low_demand": {
            "rows": int(low_mask.sum()),
            "base_rmsle": base_low["rmsle"],
            "hybrid_rmsle": hybrid_low["rmsle"],
            "improvement_pct": low_improvement,
        },
        "high_demand": {
            "rows": int(high_mask.sum()),
            "base_rmsle": base_high["rmsle"],
            "hybrid_rmsle": hybrid_high["rmsle"],
            "improvement_pct": (
                (base_high["rmsle"] - hybrid_high["rmsle"]) / base_high["rmsle"] * 100
            ),
        },
    }


def save_figure(final: dict[str, object], output: Path) -> None:
    names = ["Base", "Long memory", "Guarded hybrid"]
    rmsle = [
        float(final["base"]["rmsle"]),
        float(final["long_memory"]["rmsle"]),
        float(final["hybrid"]["rmsle"]),
    ]
    mae = [
        float(final["base"]["mae"]),
        float(final["long_memory"]["mae"]),
        float(final["hybrid"]["mae"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].bar(names, rmsle, color=["#9aa8b3", "#d56b36", "#23788c"])
    axes[0].set_title("Final weeks 136-145 RMSLE")
    axes[0].set_ylabel("Lower is better")
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(names, mae, color=["#9aa8b3", "#d56b36", "#23788c"])
    axes[1].set_title("Final weeks 136-145 MAE")
    axes[1].tick_params(axis="x", rotation=15)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        for bars in axis.containers:
            axis.bar_label(bars, fmt="%.3f", padding=3)
    fig.suptitle(f"Guarded hybrid decision — {final['decision']}")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    prepared = prepare_data(args.data_dir)
    calibration_folds = [
        build_fold_predictions(prepared, cutoff, args.horizon, args.num_boost_round)
        for cutoff in CALIBRATION_CUTOFFS
    ]
    calibration = pd.concat(calibration_folds, ignore_index=True)
    calibration_result = calibrate_hybrid(calibration)
    selected = calibration_result["selected"]
    print(
        f"Selected on weeks 116-135: threshold={selected['threshold']}, "
        f"long_memory_weight={selected['long_memory_weight']}",
        flush=True,
    )

    final_scored = build_fold_predictions(
        prepared,
        FINAL_TEST_CUTOFF,
        args.horizon,
        args.num_boost_round,
    )
    final = evaluate_final(final_scored, selected)
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Use base predictions below a calibrated threshold; blend long-memory above it.",
        "calibration_week_ranges": [[116, 125], [126, 135]],
        "final_test_week_range": [136, 145],
        "num_boost_round": args.num_boost_round,
        "calibration": calibration_result,
        "final_test": final,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_figure(final, args.figure)
    print(f"Final decision: {final['decision']}", flush=True)
    print(f"Final RMSLE improvement: {final['rmsle_improvement_pct']:.3f}%", flush=True)
    print(f"Saved experiment: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
