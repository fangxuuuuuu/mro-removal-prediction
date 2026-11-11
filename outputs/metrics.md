# Results Summary

Dataset: 84,000 snapshots · 1,471 components · 40 aircraft · positive rate 1.76%

Splits: train 29,949 / validation 5,460 / test 5,880 rows (test fleet: 14 aircraft, cycles >= 4800)

| Model | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Recall @ op | Alert rate /100 | Meets brief? |
|---|---|---|---|---|---|---|
| logistic_regression | 0.788 | 0.814 | 0.997 | 100.0% | 2.77% | YES |
| xgboost | 0.794 | 0.814 | 0.997 | 98.6% | 2.87% | YES |
| hist_gradient_boosting | 0.733 | 0.757 | 0.996 | 97.3% | 2.86% | YES |

Operating point (threshold) selected on validation only: xgboost thr=0.0122
