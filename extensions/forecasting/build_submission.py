"""Train on all labeled weeks and build a real Kaggle test submission."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pipeline import format_submission, prepare_data, recursive_predict, train_point_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build recursive predictions for the original unlabeled test.csv file."
    )
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(__file__).parent / "outputs"
    parser.add_argument("--data-dir", type=Path, default=repo_root / "data")
    parser.add_argument("--output", type=Path, default=output_dir / "osman_submission.csv")
    parser.add_argument("--manifest", type=Path, default=output_dir / "osman_submission_manifest.json")
    parser.add_argument("--num-boost-round", type=int, default=1360)
    parser.add_argument("--save-model", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    prepared = prepare_data(args.data_dir)
    print(
        f"Training on weeks {int(prepared.train_observed.week.min())}-"
        f"{int(prepared.train_observed.week.max())} ({len(prepared.train_observed):,} rows)",
        flush=True,
    )
    model = train_point_model(
        prepared.train_observed,
        num_boost_round=args.num_boost_round,
    )
    predictions = recursive_predict(
        model,
        prepared.test_static,
        prepared.train_static,
    )
    submission = format_submission(prepared.sample_submission, predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)

    if args.save_model:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(args.save_model))

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_week_range": [
            int(prepared.train_observed.week.min()),
            int(prepared.train_observed.week.max()),
        ],
        "test_week_range": [
            int(prepared.test_static.week.min()),
            int(prepared.test_static.week.max()),
        ],
        "training_rows": len(prepared.train_observed),
        "submission_rows": len(submission),
        "num_boost_round": args.num_boost_round,
        "prediction_min": float(submission.num_orders.min()),
        "prediction_mean": float(submission.num_orders.mean()),
        "prediction_max": float(submission.num_orders.max()),
        "weekly_prediction_summary": [
            {
                "week": int(week),
                "rows": len(week_frame),
                "prediction_min": float(week_frame.num_orders.min()),
                "prediction_mean": float(week_frame.num_orders.mean()),
                "prediction_max": float(week_frame.num_orders.max()),
            }
            for week, week_frame in predictions.groupby("week", sort=True)
        ],
        "recursive_features": [
            "num_orders_lag_1",
            "num_orders_lag_4",
            "num_orders_roll_mean_4",
        ],
        "submission_path": (
            args.output.resolve().relative_to(repo_root).as_posix()
            if args.output.resolve().is_relative_to(repo_root)
            else str(args.output.resolve())
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved submission: {args.output.resolve()} ({len(submission):,} rows)", flush=True)
    print(f"Saved manifest: {args.manifest.resolve()}", flush=True)


if __name__ == "__main__":
    main()
