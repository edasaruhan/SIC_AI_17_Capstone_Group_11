"""Run expanding-window, recursive walk-forward validation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pipeline import (
    prepare_data,
    recursive_persistence_predict,
    recursive_predict,
    regression_metrics,
    train_point_model,
)


DEFAULT_CUTOFFS = [115, 125, 135]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-aware expanding-window backtest using recursive multi-week forecasts."
    )
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--data-dir", type=Path, default=repo_root / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs" / "osman_walk_forward_metrics.json")
    parser.add_argument("--cutoffs", type=int, nargs="+", default=DEFAULT_CUTOFFS)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--num-boost-round", type=int, default=1360)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.horizon <= 0:
        raise ValueError("horizon must be positive")
    prepared = prepare_data(args.data_dir)
    folds: list[dict[str, object]] = []

    for cutoff in args.cutoffs:
        validation_end = cutoff + args.horizon
        observed = prepared.train_static[prepared.train_static["week"] <= cutoff].copy()
        train_frame = prepared.train_observed[prepared.train_observed["week"] <= cutoff].copy()
        validation = prepared.train_static[
            (prepared.train_static["week"] > cutoff)
            & (prepared.train_static["week"] <= validation_end)
        ].copy()
        if validation.empty:
            raise ValueError(f"No validation rows found for cutoff {cutoff}")

        print(
            f"Fold cutoff={cutoff}: train_rows={len(train_frame):,}, "
            f"validation_weeks={int(validation.week.min())}-{int(validation.week.max())}, "
            f"validation_rows={len(validation):,}",
            flush=True,
        )
        model = train_point_model(train_frame, num_boost_round=args.num_boost_round)
        predictions = recursive_predict(model, validation, observed)
        persistence_predictions = recursive_persistence_predict(validation, observed)
        scored = validation[["id", "num_orders"]].merge(
            predictions[["id", "num_orders"]],
            on="id",
            suffixes=("_actual", "_predicted"),
            validate="one_to_one",
        )
        scored = scored.merge(
            persistence_predictions[["id", "num_orders"]].rename(
                columns={"num_orders": "num_orders_persistence"}
            ),
            on="id",
            validate="one_to_one",
        )
        metrics = regression_metrics(
            scored["num_orders_actual"].to_numpy(),
            scored["num_orders_predicted"].to_numpy(),
        )
        persistence_metrics = regression_metrics(
            scored["num_orders_actual"].to_numpy(),
            scored["num_orders_persistence"].to_numpy(),
        )
        scored = scored.merge(validation[["id", "week"]], on="id", validate="one_to_one")
        weekly_metrics = []
        for week, week_frame in scored.groupby("week", sort=True):
            model_week = regression_metrics(
                week_frame["num_orders_actual"].to_numpy(),
                week_frame["num_orders_predicted"].to_numpy(),
            )
            persistence_week = regression_metrics(
                week_frame["num_orders_actual"].to_numpy(),
                week_frame["num_orders_persistence"].to_numpy(),
            )
            weekly_metrics.append(
                {
                    "week": int(week),
                    "rows": len(week_frame),
                    "rmsle": model_week["rmsle"],
                    "mae": model_week["mae"],
                    "persistence_rmsle": persistence_week["rmsle"],
                    "persistence_mae": persistence_week["mae"],
                }
            )
        fold = {
            "train_end_week": cutoff,
            "validation_start_week": int(validation["week"].min()),
            "validation_end_week": int(validation["week"].max()),
            "train_rows": len(train_frame),
            "validation_rows": len(validation),
            **metrics,
            "persistence_rmsle": persistence_metrics["rmsle"],
            "persistence_mae": persistence_metrics["mae"],
            "rmsle_improvement_vs_persistence_pct": float(
                (persistence_metrics["rmsle"] - metrics["rmsle"])
                / persistence_metrics["rmsle"]
                * 100
            ),
            "weekly_metrics": weekly_metrics,
        }
        folds.append(fold)
        print(
            f"  Model RMSLE={metrics['rmsle']:.5f}, MAE={metrics['mae']:.2f}; "
            f"persistence RMSLE={persistence_metrics['rmsle']:.5f}, "
            f"improvement={fold['rmsle_improvement_vs_persistence_pct']:.2f}%",
            flush=True,
        )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation": "expanding-window recursive multi-week backtest",
        "num_boost_round": args.num_boost_round,
        "horizon_weeks": args.horizon,
        "folds": folds,
        "mean_rmsle": float(np.mean([fold["rmsle"] for fold in folds])),
        "std_rmsle": float(np.std([fold["rmsle"] for fold in folds])),
        "mean_mae": float(np.mean([fold["mae"] for fold in folds])),
        "mean_persistence_rmsle": float(
            np.mean([fold["persistence_rmsle"] for fold in folds])
        ),
        "mean_rmsle_improvement_vs_persistence_pct": float(
            np.mean([fold["rmsle_improvement_vs_persistence_pct"] for fold in folds])
        ),
        "notes": [
            "Each fold trains only on weeks at or before its cutoff.",
            "Validation lag features use prior predictions, not future observed targets.",
            "The external Kaggle test.csv is not used during model selection.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved metrics: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
