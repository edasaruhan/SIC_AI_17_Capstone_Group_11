"""Leakage-aware forecasting utilities for the capstone extension.

This module intentionally does not import or modify the original notebook scripts.
It recreates the required features from the raw CSV files and supports recursive
multi-week prediction where future demand is not available.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "checkout_price",
    "base_price",
    "discount_ratio",
    "emailer_for_promotion",
    "homepage_featured",
    "num_orders_lag_1",
    "num_orders_lag_4",
    "num_orders_roll_mean_4",
    "weekofyear",
    "week_sin",
    "week_cos",
    "center_type_enc",
    "category_enc",
    "cuisine_enc",
    "city_code",
    "region_code",
    "op_area",
    "center_id",
    "meal_id",
]

CATEGORICAL_FEATURES = [
    "center_type_enc",
    "category_enc",
    "cuisine_enc",
    "center_id",
    "meal_id",
]

TUNED_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "verbose": -1,
    "seed": 42,
    "feature_pre_filter": False,
    "learning_rate": 0.02692053280761391,
    "num_leaves": 64,
    "min_data_in_leaf": 48,
    "feature_fraction": 0.708312282750701,
    "bagging_fraction": 0.8454887011519805,
    "bagging_freq": 4,
    "lambda_l1": 0.019201401202374752,
    "lambda_l2": 5.509697698456197e-05,
}

RAW_REQUIRED_COLUMNS = {
    "train.csv": {
        "id",
        "week",
        "center_id",
        "meal_id",
        "checkout_price",
        "base_price",
        "emailer_for_promotion",
        "homepage_featured",
        "num_orders",
    },
    "test.csv": {
        "id",
        "week",
        "center_id",
        "meal_id",
        "checkout_price",
        "base_price",
        "emailer_for_promotion",
        "homepage_featured",
    },
    "meal_info.csv": {"meal_id", "category", "cuisine"},
    "fulfilment_center_info.csv": {
        "center_id",
        "city_code",
        "region_code",
        "center_type",
        "op_area",
    },
    "sample_submission.csv": {"id", "num_orders"},
}


class Predictor(Protocol):
    def predict(self, data: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class PreparedData:
    train_static: pd.DataFrame
    train_observed: pd.DataFrame
    test_static: pd.DataFrame
    sample_submission: pd.DataFrame
    category_maps: Mapping[str, Mapping[str, int]]


def _read_required_csv(data_dir: Path, filename: str) -> pd.DataFrame:
    path = data_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required dataset is missing: {path}")
    frame = pd.read_csv(path)
    missing = RAW_REQUIRED_COLUMNS[filename] - set(frame.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")
    return frame


def _validate_raw_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    meal_info: pd.DataFrame,
    center_info: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> None:
    for name, frame in (("train", train), ("test", test), ("sample_submission", sample_submission)):
        if frame["id"].isna().any() or not frame["id"].is_unique:
            raise ValueError(f"{name} must contain unique, non-null ids")
    if set(test["id"]) != set(sample_submission["id"]):
        raise ValueError("test.csv and sample_submission.csv contain different ids")
    if meal_info["meal_id"].duplicated().any():
        raise ValueError("meal_info.csv contains duplicate meal_id values")
    if center_info["center_id"].duplicated().any():
        raise ValueError("fulfilment_center_info.csv contains duplicate center_id values")
    if train["num_orders"].isna().any() or (train["num_orders"] < 0).any():
        raise ValueError("train num_orders must be non-null and non-negative")
    for name, frame in (("train", train), ("test", test)):
        if frame.duplicated(["week", "center_id", "meal_id"]).any():
            raise ValueError(f"{name} contains duplicate center/meal/week rows")


def _merge_metadata(
    frame: pd.DataFrame,
    meal_info: pd.DataFrame,
    center_info: pd.DataFrame,
) -> pd.DataFrame:
    merged = frame.merge(meal_info, on="meal_id", how="left", validate="many_to_one")
    merged = merged.merge(center_info, on="center_id", how="left", validate="many_to_one")
    required_metadata = ["category", "cuisine", "city_code", "region_code", "center_type", "op_area"]
    if merged[required_metadata].isna().any().any():
        missing_rows = int(merged[required_metadata].isna().any(axis=1).sum())
        raise ValueError(f"Metadata join left {missing_rows} rows incomplete")
    return merged


def _fit_category_maps(frames: Iterable[pd.DataFrame]) -> dict[str, dict[str, int]]:
    combined = pd.concat(
        [frame[["center_type", "category", "cuisine"]] for frame in frames],
        ignore_index=True,
    )
    result: dict[str, dict[str, int]] = {}
    for column in ("center_type", "category", "cuisine"):
        values = sorted(combined[column].astype(str).unique().tolist())
        result[column] = {value: index for index, value in enumerate(values)}
    return result


def add_static_features(
    frame: pd.DataFrame,
    category_maps: Mapping[str, Mapping[str, int]],
) -> pd.DataFrame:
    result = frame.copy()
    result["discount_ratio"] = np.where(
        result["base_price"].ne(0),
        (result["base_price"] - result["checkout_price"]) / result["base_price"],
        np.nan,
    )
    result["weekofyear"] = ((result["week"] - 1) % 52) + 1
    result["week_sin"] = np.sin(2 * np.pi * result["weekofyear"] / 52)
    result["week_cos"] = np.cos(2 * np.pi * result["weekofyear"] / 52)
    for column, mapping in category_maps.items():
        result[f"{column}_enc"] = result[column].astype(str).map(mapping).fillna(-1).astype("int32")
    return result


def add_observed_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    if "num_orders" not in frame:
        raise ValueError("Observed lag features require num_orders")
    result = frame.sort_values(["center_id", "meal_id", "week", "id"]).reset_index(drop=True)
    grouped = result.groupby(["center_id", "meal_id"], sort=False)["num_orders"]
    result["num_orders_lag_1"] = grouped.shift(1)
    result["num_orders_lag_4"] = grouped.shift(4)
    result["num_orders_roll_mean_4"] = grouped.transform(
        lambda series: series.shift(1).rolling(4).mean()
    )
    result["log_num_orders"] = np.log1p(result["num_orders"])
    return result


def prepare_data(data_dir: Path) -> PreparedData:
    data_dir = data_dir.resolve()
    train = _read_required_csv(data_dir, "train.csv")
    test = _read_required_csv(data_dir, "test.csv")
    meal_info = _read_required_csv(data_dir, "meal_info.csv")
    center_info = _read_required_csv(data_dir, "fulfilment_center_info.csv")
    sample_submission = _read_required_csv(data_dir, "sample_submission.csv")
    _validate_raw_data(train, test, meal_info, center_info, sample_submission)

    train_merged = _merge_metadata(train, meal_info, center_info)
    test_merged = _merge_metadata(test, meal_info, center_info)
    category_maps = _fit_category_maps([train_merged, test_merged])
    train_static = add_static_features(train_merged, category_maps)
    test_static = add_static_features(test_merged, category_maps)
    train_observed = add_observed_lag_features(train_static)
    return PreparedData(
        train_static=train_static,
        train_observed=train_observed,
        test_static=test_static,
        sample_submission=sample_submission,
        category_maps=category_maps,
    )


History = dict[tuple[int, int], list[float]]


def make_history(observed: pd.DataFrame) -> History:
    required = {"center_id", "meal_id", "week", "id", "num_orders"}
    missing = required - set(observed.columns)
    if missing:
        raise ValueError(f"History data is missing columns: {sorted(missing)}")
    histories: defaultdict[tuple[int, int], list[float]] = defaultdict(list)
    ordered = observed.sort_values(["week", "center_id", "meal_id", "id"])
    for row in ordered.itertuples(index=False):
        histories[(int(row.center_id), int(row.meal_id))].append(float(row.num_orders))
    return dict(histories)


def add_history_features(batch: pd.DataFrame, histories: History) -> pd.DataFrame:
    result = batch.copy()
    lag_1: list[float] = []
    lag_4: list[float] = []
    rolling_4: list[float] = []
    for row in result.itertuples(index=False):
        history = histories.get((int(row.center_id), int(row.meal_id)), [])
        lag_1.append(history[-1] if len(history) >= 1 else np.nan)
        lag_4.append(history[-4] if len(history) >= 4 else np.nan)
        rolling_4.append(float(np.mean(history[-4:])) if len(history) >= 4 else np.nan)
    result["num_orders_lag_1"] = lag_1
    result["num_orders_lag_4"] = lag_4
    result["num_orders_roll_mean_4"] = rolling_4
    return result


def update_history(batch: pd.DataFrame, predictions: np.ndarray, histories: History) -> None:
    if len(batch) != len(predictions):
        raise ValueError("Batch and prediction lengths do not match")
    for row, prediction in zip(batch.itertuples(index=False), predictions, strict=True):
        key = (int(row.center_id), int(row.meal_id))
        histories.setdefault(key, []).append(float(prediction))


def recursive_predict(
    model: Predictor,
    future_static: pd.DataFrame,
    observed_history: pd.DataFrame,
) -> pd.DataFrame:
    if future_static.empty:
        raise ValueError("Future dataset is empty")
    histories = make_history(observed_history)
    prediction_parts: list[pd.DataFrame] = []
    for week in sorted(future_static["week"].unique().tolist()):
        batch = future_static[future_static["week"] == week].copy()
        batch = batch.sort_values(["center_id", "meal_id", "id"]).reset_index(drop=True)
        featured = add_history_features(batch, histories)
        prediction_log = np.asarray(model.predict(featured[FEATURE_COLUMNS]), dtype=float)
        predictions = np.clip(np.expm1(prediction_log), 0, None)
        if not np.isfinite(predictions).all():
            raise ValueError(f"Non-finite predictions produced for week {week}")
        update_history(batch, predictions, histories)
        prediction_parts.append(
            featured[["id", "week", "center_id", "meal_id"]].assign(num_orders=predictions)
        )
    return pd.concat(prediction_parts, ignore_index=True)


def recursive_persistence_predict(
    future_static: pd.DataFrame,
    observed_history: pd.DataFrame,
) -> pd.DataFrame:
    """Forecast each series with its latest known value, recursively by week."""
    if future_static.empty:
        raise ValueError("Future dataset is empty")
    histories = make_history(observed_history)
    fallback = float(observed_history["num_orders"].median())
    prediction_parts: list[pd.DataFrame] = []
    for week in sorted(future_static["week"].unique().tolist()):
        batch = future_static[future_static["week"] == week].copy()
        batch = batch.sort_values(["center_id", "meal_id", "id"]).reset_index(drop=True)
        predictions = np.asarray(
            [
                histories.get((int(row.center_id), int(row.meal_id)), [fallback])[-1]
                for row in batch.itertuples(index=False)
            ],
            dtype=float,
        )
        update_history(batch, predictions, histories)
        prediction_parts.append(
            batch[["id", "week", "center_id", "meal_id"]].assign(num_orders=predictions)
        )
    return pd.concat(prediction_parts, ignore_index=True)


def train_point_model(
    train_observed: pd.DataFrame,
    num_boost_round: int = 1360,
    params: Mapping[str, object] | None = None,
) -> lgb.Booster:
    if num_boost_round <= 0:
        raise ValueError("num_boost_round must be positive")
    training_params = dict(TUNED_PARAMS)
    if params:
        training_params.update(params)
    dataset = lgb.Dataset(
        train_observed[FEATURE_COLUMNS],
        label=train_observed["log_num_orders"],
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False,
    )
    return lgb.train(training_params, dataset, num_boost_round=num_boost_round)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.clip(np.asarray(predicted, dtype=float), 0, None)
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted arrays must have the same shape")
    return {
        "rmsle": float(np.sqrt(np.mean((np.log1p(actual) - np.log1p(predicted)) ** 2))),
        "mae": float(np.mean(np.abs(actual - predicted))),
    }


def format_submission(
    sample_submission: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    if predictions["id"].duplicated().any():
        raise ValueError("Predictions contain duplicate ids")
    result = sample_submission[["id"]].merge(
        predictions[["id", "num_orders"]],
        on="id",
        how="left",
        validate="one_to_one",
    )
    if result["num_orders"].isna().any():
        raise ValueError("Submission is missing predictions for one or more ids")
    if len(result) != len(sample_submission) or (result["num_orders"] < 0).any():
        raise ValueError("Submission failed row-count or non-negative prediction validation")
    return result


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
