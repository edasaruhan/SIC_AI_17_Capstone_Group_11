from __future__ import annotations

import numpy as np
import pandas as pd

from monitoring.drift_report import compute_drift, psi
from monitoring.performance_report import compute_performance


def test_psi_distinguishes_stable_and_shifted_samples():
    reference = np.linspace(10, 100, 500)
    stable = reference.copy()
    shifted = reference * 1.8
    assert psi(reference, stable) < 0.01
    assert psi(reference, shifted) > 0.25


def test_compute_drift_reports_ok_for_matching_distribution():
    reference = pd.DataFrame(
        {
            "checkout_price": np.linspace(100, 200, 300),
            "base_price": np.linspace(120, 220, 300),
            "emailer_for_promotion": np.tile([0, 1], 150),
            "homepage_featured": np.tile([0, 0, 1], 100),
            "pred_p50": np.linspace(50, 500, 300),
        }
    )
    current = reference.rename(columns={"pred_p50": "pred"}).copy()
    current["n_history"] = 8
    result = compute_drift(reference, current)
    assert result["status"] == "OK"
    assert result["cold_start_share"] == 0


def test_performance_reports_waste_oriented_metrics():
    count = 10
    predictions = pd.DataFrame(
        {
            "week": np.arange(1, count + 1),
            "center_id": [1] * count,
            "meal_id": [2] * count,
            "pred": [100, 105, 95, 100, 110, 90, 100, 100, 130, 70],
            "p_low": [80] * 8 + [110, 110],
            "p_high": [120] * count,
            "n_history": [8] * count,
        }
    )
    actuals = pd.DataFrame(
        {
            "week": np.arange(1, count + 1),
            "center_id": [1] * count,
            "meal_id": [2] * count,
            "num_orders": [100] * count,
        }
    )
    result = compute_performance(predictions, actuals)
    assert result["n_matched"] == count
    assert result["wape"] == 0.09
    assert result["overforecast_rate"] == 0.3
    assert result["underforecast_rate"] == 0.3
    assert result["coverage"] == 0.8
    assert result["status"] == "OK"
