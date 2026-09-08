# Forecasting Extension

This folder adds leakage-aware evaluation and real test-set submission tooling without changing the original notebooks, reports, model files, or result files.

## What this adds

- Expanding-window walk-forward validation over three 10-week folds.
- Comparison against a recursive last-observation persistence baseline.
- Week-level backtest metrics to reveal error drift across the forecast horizon.
- Recursive lag generation: future validation and Kaggle test rows use earlier predictions instead of unavailable future targets.
- Full-data training on weeks 1-145.
- Prediction for the original unlabeled `test.csv` covering weeks 146-155.
- Submission validation against `sample_submission.csv`.
- Small unit tests for lag timing, recursive history, metrics, and submission order.

## Data and evaluation contract

- Grain: one `center_id` and `meal_id` observation per available week.
- Target: non-negative `num_orders`.
- Training features may only use demand observed before the prediction row.
- Backtest validation features use prior predictions after each fold cutoff.
- The Kaggle test set is never used for model selection or metric reporting because it has no labels.
- Default walk-forward folds train through weeks 115, 125, and 135 and validate the following 10 weeks.

## Run tests

From the repository root:

```powershell
python -m unittest discover -s extensions/forecasting/tests -v
```

## Run walk-forward validation

```powershell
python extensions/forecasting/walk_forward_validation.py
```

For a quick smoke run:

```powershell
python extensions/forecasting/walk_forward_validation.py --cutoffs 135 --num-boost-round 50
```

## Build the real submission

```powershell
python extensions/forecasting/build_submission.py
```

Generated files are placed under `extensions/forecasting/outputs/` and are intentionally ignored by Git. The default submission uses the tuned LightGBM parameters already reported by the project and 1,360 boosting rounds.

## Important interpretation

The recursive backtest is deliberately harder than the earlier teacher-forced holdout evaluation. Its metrics may therefore be worse, but they better represent a 10-week static submission where future `num_orders` values are unavailable.
