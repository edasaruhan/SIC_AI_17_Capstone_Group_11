# Forecasting Experiment Log

This log records model decisions made with the recursive validation pipeline. The
scores below are stricter than the project's reported `0.47534` holdout RMSLE
because each validation week's lag features are built from earlier predictions,
not from future observed targets. They should not be compared as if they used the
same protocol.

| Experiment | Mean/final RMSLE | Result | Reason |
| --- | ---: | --- | --- |
| Recursive base model, three folds | 0.49833 | Keep as reference | Improves over persistence by 38.32% on average. |
| Promotion interaction features, three folds | 0.49845 | Reject | Mean RMSLE worsened by 0.02%, and one fold regressed by 0.33%. |
| Long-memory demand features, three folds | 0.49234 | Reject | Mean RMSLE improved by 1.20%, but latest-fold low-demand RMSLE worsened by 1.99%. |
| Calibrated base/long-memory hybrid, final weeks 136-145 | 0.49285 | Reject | Overall RMSLE improved by 1.78% and MAE improved, but low-demand RMSLE worsened by 0.65%, exceeding the predeclared 0.50% guardrail. |

The default feature set remains unchanged. Rejected candidates are available only
through explicit feature-column arguments in the experiment scripts; they are not
used by the submission builder.
