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
- Slice-based error analysis by week, category, cuisine, center type, promotion, demand, and discount.

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

## Run error analysis

```powershell
python extensions/forecasting/error_analysis.py
```

This writes a JSON report and a four-panel PNG under `extensions/forecasting/outputs/`.

## Run the guarded feature experiment

Run the baseline walk-forward command first, then:

```powershell
python extensions/forecasting/feature_experiment.py
```

Candidate sets can test promotion-price interactions, longer demand memory, or both. A candidate is promoted only when mean RMSLE improves by at least 0.2%, the worst fold does not worsen, no individual fold regresses by more than 0.2%, and the latest low-demand slice does not regress by more than 0.5%.

```powershell
python extensions/forecasting/feature_experiment.py --candidate-set promotion
python extensions/forecasting/feature_experiment.py --candidate-set long-memory
python extensions/forecasting/feature_experiment.py --candidate-set combined
```

## Run the guarded hybrid experiment

```powershell
python extensions/forecasting/hybrid_experiment.py
```

The hybrid threshold and blend weight are selected only on weeks 116-135. Weeks 136-145 are then evaluated once as a final holdout. Low-demand RMSLE and MAE are mandatory promotion guardrails because overforecasting weak demand can increase waste.

## Build the real submission

```powershell
python extensions/forecasting/build_submission.py
```

Generated files are placed under `extensions/forecasting/outputs/` and are intentionally ignored by Git. The default submission uses the tuned LightGBM parameters already reported by the project and 1,360 boosting rounds.

## Important interpretation

The recursive backtest is deliberately harder than the earlier teacher-forced holdout evaluation. Its metrics may therefore be worse, but they better represent a 10-week static submission where future `num_orders` values are unavailable.
