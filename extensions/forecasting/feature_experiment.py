"""Evaluate a focused feature candidate behind an automatic promotion gate."""

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
    EXPERIMENTAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    LONG_MEMORY_FEATURE_COLUMNS,
    PROMOTION_FEATURE_COLUMNS,
    prepare_data,
    recursive_predict,
    regression_metrics,
    train_point_model,
)


DEFAULT_CUTOFFS = [115, 125, 135]
MINIMUM_MEAN_IMPROVEMENT_PCT = 0.2
MAXIMUM_FOLD_REGRESSION_PCT = 0.2
MAXIMUM_LOW_DEMAND_REGRESSION_PCT = 0.5

FEATURE_SETS = {
    "promotion": PROMOTION_FEATURE_COLUMNS,
    "long-memory": LONG_MEMORY_FEATURE_COLUMNS,
    "combined": EXPERIMENTAL_FEATURE_COLUMNS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate long-memory and promotion features.")
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(__file__).parent / "outputs"
    parser.add_argument("--data-dir", type=Path, default=repo_root / "data")
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=output_dir / "osman_walk_forward_metrics.json",
    )
    parser.add_argument(
        "--error-analysis",
        type=Path,
        default=output_dir / "osman_error_analysis.json",
    )
    parser.add_argument("--candidate-set", choices=sorted(FEATURE_SETS), default="promotion")
    parser.add_argument("--output", type=Path, default=output_dir / "osman_feature_experiment.json")
    parser.add_argument("--figure", type=Path, default=output_dir / "osman_feature_experiment.png")
    parser.add_argument("--cutoffs", type=int, nargs="+", default=DEFAULT_CUTOFFS)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--num-boost-round", type=int, default=1360)
    return parser.parse_args()


def load_baseline_metrics(path: Path, args: argparse.Namespace) -> dict[int, dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Baseline metrics not found at {path}. Run walk_forward_validation.py first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("num_boost_round") != args.num_boost_round:
        raise ValueError("Baseline and candidate must use the same num_boost_round")
    if payload.get("horizon_weeks") != args.horizon:
        raise ValueError("Baseline and candidate must use the same forecast horizon")
    folds = {int(fold["train_end_week"]): fold for fold in payload["folds"]}
    if set(folds) != set(args.cutoffs):
        raise ValueError("Baseline and candidate must use the same fold cutoffs")
    return folds


def slice_metrics(scored: pd.DataFrame, mask: pd.Series) -> dict[str, float | int]:
    subset = scored[mask]
    metrics = regression_metrics(
        subset["actual"].to_numpy(),
        subset["predicted"].to_numpy(),
    )
    return {"rows": len(subset), **metrics}


def promotion_decision(
    baseline_folds: dict[int, dict[str, object]],
    candidate_folds: list[dict[str, object]],
    baseline_latest_low_rmsle: float,
) -> dict[str, object]:
    baseline_values = np.asarray(
        [float(baseline_folds[int(fold["train_end_week"])]["rmsle"]) for fold in candidate_folds]
    )
    candidate_values = np.asarray([float(fold["rmsle"]) for fold in candidate_folds])
    fold_changes_pct = (baseline_values - candidate_values) / baseline_values * 100
    mean_improvement_pct = float(
        (baseline_values.mean() - candidate_values.mean()) / baseline_values.mean() * 100
    )
    latest_candidate = max(candidate_folds, key=lambda fold: int(fold["train_end_week"]))
    candidate_latest_low_rmsle = float(latest_candidate["low_demand"]["rmsle"])
    low_demand_change_pct = float(
        (baseline_latest_low_rmsle - candidate_latest_low_rmsle)
        / baseline_latest_low_rmsle
        * 100
    )
    gates = {
        "mean_improvement_at_least_0_2_pct": mean_improvement_pct
        >= MINIMUM_MEAN_IMPROVEMENT_PCT,
        "worst_fold_not_worse": float(candidate_values.max()) <= float(baseline_values.max()),
        "no_fold_regresses_more_than_0_2_pct": bool(
            np.all(fold_changes_pct >= -MAXIMUM_FOLD_REGRESSION_PCT)
        ),
        "latest_low_demand_regression_within_0_5_pct": low_demand_change_pct
        >= -MAXIMUM_LOW_DEMAND_REGRESSION_PCT,
    }
    return {
        "decision": "PROMOTE" if all(gates.values()) else "REJECT",
        "gates": gates,
        "baseline_mean_rmsle": float(baseline_values.mean()),
        "candidate_mean_rmsle": float(candidate_values.mean()),
        "mean_improvement_pct": mean_improvement_pct,
        "baseline_worst_fold_rmsle": float(baseline_values.max()),
        "candidate_worst_fold_rmsle": float(candidate_values.max()),
        "fold_improvement_pct": fold_changes_pct.tolist(),
        "baseline_latest_low_demand_rmsle": baseline_latest_low_rmsle,
        "candidate_latest_low_demand_rmsle": candidate_latest_low_rmsle,
        "latest_low_demand_improvement_pct": low_demand_change_pct,
    }


def save_figure(
    baseline_folds: dict[int, dict[str, object]],
    candidate_folds: list[dict[str, object]],
    decision: dict[str, object],
    output: Path,
) -> None:
    cutoffs = [int(fold["train_end_week"]) for fold in candidate_folds]
    baseline = [float(baseline_folds[cutoff]["rmsle"]) for cutoff in cutoffs]
    candidate = [float(fold["rmsle"]) for fold in candidate_folds]
    labels = [f"{cutoff + 1}-{cutoff + 10}" for cutoff in cutoffs]
    positions = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(positions - width / 2, baseline, width, label="Current features", color="#9aa8b3")
    ax.bar(positions + width / 2, candidate, width, label="Candidate features", color="#23788c")
    ax.set_xticks(positions, labels)
    ax.set_xlabel("Validation weeks")
    ax.set_ylabel("Recursive RMSLE (lower is better)")
    ax.set_title(f"Feature experiment — {decision['decision']}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.4f", padding=3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    baseline_folds = load_baseline_metrics(args.baseline_metrics, args)
    if not args.error_analysis.is_file():
        raise FileNotFoundError(
            f"Error analysis not found at {args.error_analysis}. Run error_analysis.py first."
        )
    error_analysis = json.loads(args.error_analysis.read_text(encoding="utf-8"))
    low_demand_rows = [
        row for row in error_analysis["slices"]["demand_bucket"] if row["value"] == "0-50"
    ]
    if len(low_demand_rows) != 1:
        raise ValueError("Baseline error analysis does not contain the 0-50 demand slice")
    baseline_latest_low_rmsle = float(low_demand_rows[0]["rmsle"])
    feature_columns = FEATURE_SETS[args.candidate_set]
    prepared = prepare_data(args.data_dir)
    candidate_folds: list[dict[str, object]] = []

    for cutoff in args.cutoffs:
        validation_end = cutoff + args.horizon
        observed = prepared.train_static[prepared.train_static["week"] <= cutoff].copy()
        train_frame = prepared.train_observed[prepared.train_observed["week"] <= cutoff].copy()
        validation = prepared.train_static[
            (prepared.train_static["week"] > cutoff)
            & (prepared.train_static["week"] <= validation_end)
        ].copy()
        print(f"Candidate fold {cutoff}: training {len(train_frame):,} rows", flush=True)
        model = train_point_model(
            train_frame,
            num_boost_round=args.num_boost_round,
            feature_columns=feature_columns,
        )
        predictions = recursive_predict(
            model,
            validation,
            observed,
            feature_columns=feature_columns,
        )
        scored = validation[["id", "num_orders"]].merge(
            predictions[["id", "num_orders"]],
            on="id",
            suffixes=("_actual", "_predicted"),
            validate="one_to_one",
        ).rename(
            columns={"num_orders_actual": "actual", "num_orders_predicted": "predicted"}
        )
        metrics = regression_metrics(scored["actual"].to_numpy(), scored["predicted"].to_numpy())
        fold_result = {
            "train_end_week": cutoff,
            "validation_start_week": int(validation.week.min()),
            "validation_end_week": int(validation.week.max()),
            "rows": len(scored),
            **metrics,
            "low_demand": slice_metrics(scored, scored["actual"] <= 50),
            "high_demand": slice_metrics(scored, scored["actual"] > 600),
        }
        candidate_folds.append(fold_result)
        baseline_rmsle = float(baseline_folds[cutoff]["rmsle"])
        improvement = (baseline_rmsle - metrics["rmsle"]) / baseline_rmsle * 100
        print(
            f"  baseline={baseline_rmsle:.5f}, candidate={metrics['rmsle']:.5f}, "
            f"change={improvement:.2f}%",
            flush=True,
        )

    decision = promotion_decision(
        baseline_folds,
        candidate_folds,
        baseline_latest_low_rmsle,
    )
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "Longer demand history and promotion-discount interaction reduce recursive drift, "
            "especially for high-demand and promoted rows."
        ),
        "candidate_set": args.candidate_set,
        "candidate_features": [
            feature
            for feature in feature_columns
            if feature not in FEATURE_COLUMNS
        ],
        "candidate_feature_count": len(feature_columns),
        "folds": candidate_folds,
        "promotion": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_figure(baseline_folds, candidate_folds, decision, args.figure)
    print(f"Decision: {decision['decision']}", flush=True)
    print(f"Mean RMSLE improvement: {decision['mean_improvement_pct']:.3f}%", flush=True)
    print(f"Saved experiment: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
