from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"
RESULTS = ROOT / "reports" / "results"
SUBMISSIONS = ROOT / "submissions"


def save_walk_forward_figure() -> None:
    metrics = json.loads((RESULTS / "walk_forward_metrics.json").read_text(encoding="utf-8"))
    labels = [
        f"{fold['validation_start_week']}-{fold['validation_end_week']}"
        for fold in metrics["folds"]
    ]
    model = [fold["rmsle"] for fold in metrics["folds"]]
    persistence = [fold["persistence_rmsle"] for fold in metrics["folds"]]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    model_bars = ax.bar(x - width / 2, model, width, label="LightGBM recursive", color="#24556a")
    persistence_bars = ax.bar(
        x + width / 2,
        persistence,
        width,
        label="Persistence baseline",
        color="#9aa8b3",
    )
    ax.set_ylabel("RMSLE")
    ax.set_xlabel("Validation weeks")
    ax.set_title("Recursive Walk Forward Validation")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.95)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.bar_label(model_bars, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(persistence_bars, fmt="%.3f", padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "final_walk_forward_validation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_submission_figure() -> None:
    manifest = json.loads(
        (SUBMISSIONS / "final_submission_manifest.json").read_text(encoding="utf-8")
    )
    weekly = manifest["weekly_prediction_summary"]
    weeks = [item["week"] for item in weekly]
    means = [item["prediction_mean"] for item in weekly]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot(weeks, means, color="#24556a", marker="o", linewidth=2.2)
    ax.fill_between(weeks, means, color="#24556a", alpha=0.12)
    for week, mean in zip(weeks, means, strict=True):
        ax.annotate(f"{mean:.0f}", (week, mean), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set_title("Final Submission Weekly Mean Demand")
    ax.set_xlabel("Week")
    ax.set_ylabel("Mean predicted orders")
    ax.set_xticks(weeks)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "final_submission_weekly_mean.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    save_walk_forward_figure()
    save_submission_figure()
    print("Final report figures generated.")
