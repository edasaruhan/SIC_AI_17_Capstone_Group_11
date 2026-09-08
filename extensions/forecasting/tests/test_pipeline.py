from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from pipeline import (  # noqa: E402
    add_history_features,
    add_observed_lag_features,
    format_submission,
    make_history,
    recursive_persistence_predict,
    regression_metrics,
    update_history,
)
from error_analysis import add_analysis_slices, summarize_slice  # noqa: E402
from hybrid_experiment import hybrid_predictions  # noqa: E402


class PipelineTests(unittest.TestCase):
    def test_observed_lags_use_only_prior_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "week": [1, 2, 3, 4, 5],
                "center_id": [10] * 5,
                "meal_id": [20] * 5,
                "num_orders": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        result = add_observed_lag_features(frame)
        self.assertTrue(np.isnan(result.loc[0, "num_orders_lag_1"]))
        self.assertEqual(result.loc[4, "num_orders_lag_1"], 40.0)
        self.assertEqual(result.loc[4, "num_orders_lag_4"], 10.0)
        self.assertEqual(result.loc[4, "num_orders_roll_mean_4"], 25.0)

    def test_recursive_history_uses_previous_prediction(self) -> None:
        observed = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "week": [1, 2, 3, 4],
                "center_id": [10] * 4,
                "meal_id": [20] * 4,
                "num_orders": [10.0, 20.0, 30.0, 40.0],
            }
        )
        histories = make_history(observed)
        week_5 = pd.DataFrame({"id": [5], "week": [5], "center_id": [10], "meal_id": [20]})
        features_5 = add_history_features(week_5, histories)
        self.assertEqual(features_5.loc[0, "num_orders_lag_1"], 40.0)
        self.assertEqual(features_5.loc[0, "num_orders_lag_4"], 10.0)
        self.assertEqual(features_5.loc[0, "num_orders_roll_mean_4"], 25.0)

        update_history(week_5, np.array([50.0]), histories)
        week_6 = pd.DataFrame({"id": [6], "week": [6], "center_id": [10], "meal_id": [20]})
        features_6 = add_history_features(week_6, histories)
        self.assertEqual(features_6.loc[0, "num_orders_lag_1"], 50.0)
        self.assertEqual(features_6.loc[0, "num_orders_lag_4"], 20.0)
        self.assertEqual(features_6.loc[0, "num_orders_roll_mean_4"], 35.0)

    def test_long_history_features_match_observed_features(self) -> None:
        observed = pd.DataFrame(
            {
                "id": list(range(1, 14)),
                "week": list(range(1, 14)),
                "center_id": [10] * 13,
                "meal_id": [20] * 13,
                "num_orders": [float(value) for value in range(1, 14)],
            }
        )
        future = pd.DataFrame({"id": [14], "week": [14], "center_id": [10], "meal_id": [20]})
        recursive = add_history_features(future, make_history(observed))
        expected = {
            "num_orders_lag_8": 6.0,
            "num_orders_lag_13": 1.0,
            "num_orders_roll_mean_8": 9.5,
            "num_orders_roll_mean_13": 7.0,
            "num_orders_roll_median_8": 9.5,
        }
        extended = add_observed_lag_features(
            pd.concat([observed, future.assign(num_orders=14.0)], ignore_index=True)
        ).iloc[-1]
        for column, value in expected.items():
            self.assertAlmostEqual(recursive.loc[0, column], value)
            self.assertAlmostEqual(extended[column], value)
        self.assertAlmostEqual(
            recursive.loc[0, "num_orders_roll_std_8"],
            extended["num_orders_roll_std_8"],
        )

    def test_submission_preserves_sample_order(self) -> None:
        sample = pd.DataFrame({"id": [3, 1, 2], "num_orders": [0, 0, 0]})
        predictions = pd.DataFrame({"id": [1, 2, 3], "num_orders": [11.0, 12.0, 13.0]})
        result = format_submission(sample, predictions)
        self.assertEqual(result["id"].tolist(), [3, 1, 2])
        self.assertEqual(result["num_orders"].tolist(), [13.0, 11.0, 12.0])

    def test_persistence_baseline_is_recursive(self) -> None:
        observed = pd.DataFrame(
            {
                "id": [1, 2],
                "week": [1, 2],
                "center_id": [10, 10],
                "meal_id": [20, 20],
                "num_orders": [10.0, 25.0],
            }
        )
        future = pd.DataFrame(
            {
                "id": [3, 4],
                "week": [3, 4],
                "center_id": [10, 10],
                "meal_id": [20, 20],
            }
        )
        result = recursive_persistence_predict(future, observed)
        self.assertEqual(result["num_orders"].tolist(), [25.0, 25.0])

    def test_metrics_match_known_values(self) -> None:
        metrics = regression_metrics(np.array([10.0, 20.0]), np.array([10.0, 20.0]))
        self.assertEqual(metrics, {"rmsle": 0.0, "mae": 0.0})

    def test_error_summary_reports_direction_and_rate(self) -> None:
        frame = pd.DataFrame({"actual": [10.0, 20.0], "predicted": [5.0, 30.0]})
        summary = summarize_slice(frame)
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["mae"], 7.5)
        self.assertEqual(summary["mean_bias"], 2.5)
        self.assertEqual(summary["underforecast_rate"], 0.5)

    def test_analysis_slice_labels_promotions(self) -> None:
        frame = pd.DataFrame(
            {
                "actual": [100.0],
                "predicted": [90.0],
                "emailer_for_promotion": [1],
                "homepage_featured": [1],
                "discount_ratio": [0.2],
            }
        )
        result = add_analysis_slices(frame)
        self.assertEqual(result.loc[0, "promotion_state"], "email_and_homepage")
        self.assertEqual(str(result.loc[0, "demand_bucket"]), "51-150")
        self.assertEqual(str(result.loc[0, "discount_bucket"]), "15-30%")

    def test_hybrid_keeps_base_below_threshold_and_blends_above(self) -> None:
        base = np.array([40.0, 200.0])
        long_memory = np.array([50.0, 100.0])
        result = hybrid_predictions(base, long_memory, threshold=100.0, long_memory_weight=0.5)
        np.testing.assert_allclose(result, np.array([40.0, 150.0]))


if __name__ == "__main__":
    unittest.main()
