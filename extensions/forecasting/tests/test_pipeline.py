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


if __name__ == "__main__":
    unittest.main()
