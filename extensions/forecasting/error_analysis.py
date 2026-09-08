"""Identify where the recursive forecast succeeds and fails."""

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

from pipeline import prepare_data, recursive_predict, regression_metrics, train_point_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run slice-based recursive forecast error analysis.")
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(__file__).parent / "outputs"
    parser.add_argument("--data-dir", type=Path, default=repo_root / "data")
    parser.add_argument("--train-end-week", type=int, default=135)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--num-boost-round", type=int, default=1360)
    parser.add_argument("--output", type=Path, default=output_dir / "osman_error_analysis.json")
    parser.add_argument("--figure", type=Path, default=output_dir / "osman_error_analysis.png")
    return parser.parse_args()


def summarize_slice(frame: pd.DataFrame) -> dict[str, float | int]:
    actual = frame["actual"].to_numpy(dtype=float)
    predicted = frame["predicted"].to_numpy(dtype=float)
    metrics = regression_metrics(actual, predicted)
    residual = predicted - actual
    actual_sum = float(actual.sum())
    return {
        "rows": len(frame),
        "rmsle": metrics["rmsle"],
        "mae": metrics["mae"],
        "wape": float(np.abs(residual).sum() / actual_sum) if actual_sum else 0.0,
        "mean_bias": float(residual.mean()),
        "underforecast_rate": float(np.mean(predicted < actual)),
    }


def summarize_dimension(frame: pd.DataFrame, dimension: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for value, group in frame.groupby(dimension, observed=True, dropna=False, sort=True):
        rows.append({"value": str(value), **summarize_slice(group)})
    return sorted(rows, key=lambda row: float(row["rmsle"]), reverse=True)


def add_analysis_slices(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["promotion_state"] = np.select(
        [
            result["emailer_for_promotion"].eq(1) & result["homepage_featured"].eq(1),
            result["emailer_for_promotion"].eq(1),
            result["homepage_featured"].eq(1),
        ],
        ["email_and_homepage", "email_only", "homepage_only"],
        default="none",
    )
    result["demand_bucket"] = pd.cut(
        result["actual"],
        bins=[-np.inf, 50, 150, 300, 600, np.inf],
        labels=["0-50", "51-150", "151-300", "301-600", "601+"],
    )
    result["discount_bucket"] = pd.cut(
        result["discount_ratio"],
        bins=[-np.inf, 0, 0.05, 0.15, 0.30, np.inf],
        labels=["none_or_markup", "0-5%", "5-15%", "15-30%", "30%+"],
    )
    return result


def _slice_frame(slices: dict[str, list[dict[str, object]]], name: str) -> pd.DataFrame:
    frame = pd.DataFrame(slices[name])
    return frame.sort_values("rmsle", ascending=False)


def save_figure(slices: dict[str, list[dict[str, object]]], output: Path) -> None:
    weekly = pd.DataFrame(slices["week"]).sort_values("value")
    weekly["week_number"] = weekly["value"].astype(int)
    category = _slice_frame(slices, "category")
    demand = pd.DataFrame(slices["demand_bucket"])
    promotion = pd.DataFrame(slices["promotion_state"]).sort_values("underforecast_rate")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(weekly["week_number"], weekly["rmsle"], marker="o", color="#23788c")
    axes[0, 0].set(title="Recursive RMSLE by week", xlabel="Week", ylabel="RMSLE")
    axes[0, 0].grid(alpha=0.25)

    top_categories = category.head(8).sort_values("rmsle")
    axes[0, 1].barh(top_categories["value"], top_categories["rmsle"], color="#d56b36")
    axes[0, 1].set(title="Highest-error categories", xlabel="RMSLE")

    demand_order = ["0-50", "51-150", "151-300", "301-600", "601+"]
    demand["value"] = pd.Categorical(demand["value"], categories=demand_order, ordered=True)
    demand = demand.sort_values("value")
    axes[1, 0].bar(demand["value"].astype(str), demand["rmsle"], color="#4b9b6f")
    axes[1, 0].set(title="Error by actual demand", xlabel="Orders", ylabel="RMSLE")
    axes[1, 0].tick_params(axis="x", rotation=20)

    axes[1, 1].barh(
        promotion["value"], promotion["underforecast_rate"] * 100, color="#8064a2"
    )
    axes[1, 1].set(title="Underforecast rate by promotion", xlabel="Underforecast rate (%)")

    fig.suptitle("Food Demand Forecast Error Analysis — Weeks 136-145", fontsize=15)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    prepared = prepare_data(args.data_dir)
    validation_end = args.train_end_week + args.horizon
    observed = prepared.train_static[prepared.train_static["week"] <= args.train_end_week].copy()
    train_frame = prepared.train_observed[
        prepared.train_observed["week"] <= args.train_end_week
    ].copy()
    validation = prepared.train_static[
        (prepared.train_static["week"] > args.train_end_week)
        & (prepared.train_static["week"] <= validation_end)
    ].copy()
    if validation.empty:
        raise ValueError("Selected error-analysis validation window is empty")

    print("Training baseline extension model for error analysis...", flush=True)
    model = train_point_model(train_frame, num_boost_round=args.num_boost_round)
    predictions = recursive_predict(model, validation, observed)
    scored = validation.merge(
        predictions[["id", "num_orders"]].rename(columns={"num_orders": "predicted"}),
        on="id",
        validate="one_to_one",
    ).rename(columns={"num_orders": "actual"})
    scored = add_analysis_slices(scored)

    dimensions = [
        "week",
        "category",
        "cuisine",
        "center_type",
        "promotion_state",
        "demand_bucket",
        "discount_bucket",
    ]
    slices = {dimension: summarize_dimension(scored, dimension) for dimension in dimensions}
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_end_week": args.train_end_week,
        "validation_week_range": [int(validation.week.min()), int(validation.week.max())],
        "num_boost_round": args.num_boost_round,
        "overall": summarize_slice(scored),
        "slices": slices,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_figure(slices, args.figure)
    print(f"Saved analysis: {args.output.resolve()}", flush=True)
    print(f"Saved figure: {args.figure.resolve()}", flush=True)


if __name__ == "__main__":
    main()
