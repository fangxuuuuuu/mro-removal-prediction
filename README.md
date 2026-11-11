# MRO Unscheduled Component Removal Prediction — PoC

Predicts whether an aircraft component will require an **unscheduled removal
within the next 30 flight cycles**, produces a 0–100 risk score per component,
and explains the drivers behind each alert.

Full methodology, results and production design: [docs/REPORT.md](docs/REPORT.md).

## The brief (operating point)

- Detect **≥ 80%** of actual unscheduled removals
- Generate **≤ 5 alerts per 100 active components** per review window
- Compare ≥ 2 models, explain high-risk predictions, prevent temporal /
  aircraft-level / component-level leakage

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# end-to-end: generate mock data, train, evaluate, explain
python src/run_all.py

# reuse the shipped dataset instead of regenerating
python src/run_all.py --no-generate

# tests (leakage, as-of contract, alert budget)
pytest tests/ -v
```

Outputs land in `outputs/`: `metrics.md` (model comparison + operating point),
`test_predictions.csv` (per-snapshot risk scores and bands),
`example_alerts.md` (alert cards with SHAP-based explanations).

## Repository map

| Path | Purpose |
|---|---|
| `src/config.py` | Shared data dictionary (type specs, sensor limits, horizon) |
| `src/generate_data.py` | Synthetic mock dataset generator (~2% positives) |
| `src/prepare.py` | Cleaning, dtypes, physical-limit clipping |
| `src/features.py` | As-of feature engineering (trends, ratios, indicators) |
| `src/split.py` | Temporal + aircraft/component-disjoint split |
| `src/train.py` | Logistic Regression vs XGBoost vs HistGB |
| `src/evaluate.py` | Metrics, threshold selection, alert budget |
| `src/explain.py` | Global + local (TreeSHAP) explanations, alert cards |
| `src/run_all.py` | End-to-end pipeline |
| `tests/` | pytest suite |

## Mock dataset

`data/component_snapshots.csv` — one row per component observation per
checkpoint (every 20 flight cycles). Schema and generation logic are
described in [docs/REPORT.md](docs/REPORT.md) §1 and
[data/DATA_DICTIONARY.md](data/DATA_DICTIONARY.md). A packaged copy (CSV +
data dictionary) is provided via Google Drive:
[Download dataset](https://drive.google.com/file/d/1_sMO_Ce7rHH3VZfni8CRwUk4ODPWMjPh/view?usp=sharing)

You can also regenerate it deterministically with `python src/run_all.py`.

## Design highlights

- **Leakage control:** whole-aircraft hold-out for test, temporal windows,
  component instances alive at the validation boundary excluded from training,
  strictly as-of features (enforced by tests).
- **Imbalance:** cost-sensitive learning + threshold tuning — no SMOTE.
- **Threshold:** alert-rate budget is a hard constraint evaluated on the
  operational view (latest snapshot per active component); recall is maximized
  within budget, tuned on validation only.
- **Explanations:** every alert ships with its top SHAP factors.

## Limitations

See [docs/REPORT.md](docs/REPORT.md) §7 (synthetic data, label censoring,
repair effects, calibration, fleet-level effects).
