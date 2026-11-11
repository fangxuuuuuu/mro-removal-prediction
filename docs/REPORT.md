# Unscheduled Component Removal Prediction — PoC Report

**Problem:** Predict whether an aircraft component will require an **unscheduled removal within the next 30 flight cycles**, so the MRO planning team can proactively stage parts and schedule removals during planned maintenance windows instead of reacting to AOG events.

**Business constraint (from the assessment brief):** the operational alert budget is **at most 5 alerts per 100 active components per review window**, while detecting **at least 80% of actual removals**.

All numbers in this report are produced by the code in this repository against a synthetic mock dataset (see "Mock Dataset"). The pipeline is deterministic (fixed seed), so every figure is reproducible.

---

## 1. Data Preparation

### 1.1 Dataset schema

Each row of `data/component_snapshots.csv` is one **component observation** (snapshot) taken at a flight-cycle checkpoint — i.e. "component X on aircraft Y, as of cycle N". Columns:

| Field | Type | Description |
|---|---|---|
| `snapshot_id` | str | Unique row key |
| `aircraft_id` | str | Tail number (group key) |
| `component_id` | str | Physically installed part instance (group key) |
| `component_type` | category | e.g. `APU`, `Engine_Fan`, `Hydraulic_Pump`, `Landing_Gear_Brake`, `Fuel_Pump`, `Starter_Generator` |
| `ata_chapter` | category | ATA chapter derived from type |
| `snapshot_cycle` | int | Flight cycle at snapshot |
| `cycles_since_install` | int | Age of the part instance |
| `sensor_vibration` | float | Vibration amplitude (units), has gaps |
| `sensor_oil_temp` | float | Oil temperature, has gaps |
| `sensor_pressure_delta` | float | Pressure drop across the component, has gaps |
| `fault_code_count_90d` | int | Distinct fault codes seen in last 90 days |
| `has_hard_fault` | int | Hard fault present in last 30 days |
| `days_since_last_maint` | int | Days since last scheduled maintenance |
| `repairs_last_180d` | int | Number of shop visits in last 180 days |
| `label_removed_30d` | int | **1 = unscheduled removal within next 30 cycles** (the target) |

The mock generator produces a latent per-component degradation curve (age + wear + stochastic jumps). Sensors are noisy, partially-observed views of that latent curve; unscheduled removals are triggered when degradation crosses a type-specific threshold — so there *is* real signal to learn, but it is noisy and only ~2% of snapshots are positive.

**Mock dataset:** packaged CSV + data dictionary at
[Google Drive](https://drive.google.com/file/d/1_sMO_Ce7rHH3VZfni8CRwUk4ODPWMjPh/view?usp=sharing);
the schema is documented in `data/DATA_DICTIONARY.md`.

### 1.2 Missing data

- Sensor columns have realistic gaps (~8% missing, correlated within component/periods).
- Numeric features are imputed with the **training-set median** inside the model pipeline (never using test statistics), and every sensor feature also gets an explicit **missing-indicator** column, because in this domain "sensor stopped reporting" is itself a signal.
- `HistGradientBoosting` and `XGBoost` handle NaN natively, so tree models consume the raw missing values while the linear baseline uses imputed values — both paths are leakage-safe because imputation statistics are fitted on the training fold only.

### 1.3 Outliers

- Sensor readings are validated against **physical limits** per component type (from the data dictionary); values outside limits are clipped, not dropped — an extreme reading is often exactly the precursor we want to keep.
- For the linear model, features are standardized with a `RobustScaler` (median/IQR) so residual spikes don't dominate coefficients. Tree models are left unscaled (invariant to monotone transforms).

### 1.4 Categorical features

- `component_type` and `ata_chapter` are one-hot encoded for the linear model and consumed as integer categories (with `enable_categorical=True` / native categorical support) by the tree model.
- `aircraft_id` / `component_id` are **never used as features** (they would memorize fleet identity and leak group structure); they are used only for grouping and per-part aggregation features.

### 1.5 Class imbalance (~2% positive)

- Chosen approach: **cost-sensitive learning** (`class_weight="balanced"` for LR, `scale_pos_weight` for XGBoost) + **threshold tuning on the validation set** (Section 4).
- SMOTE/oversampling was deliberately avoided: it fabricates synthetic positives, distorts calibrated probabilities, and is risky near temporal boundaries (a synthetic hybrid of two snapshots can leak future information). With a tunable threshold, re-sampling is unnecessary and calibration is more valuable for MRO prioritization.

---

## 2. Data Splitting (leakage prevention)

Three leakage channels exist in this domain, and each is handled explicitly:

1. **Temporal leakage** — training on snapshots from the same period as the events being scored.
2. **Aircraft-level leakage** — the same tail appearing in train and test (fleet-level habits, routes, maintenance programs correlate).
3. **Component-level leakage** — the same physical part instance appearing on both sides: its snapshots are highly autocorrelated, so the model can "recognize" the part rather than learn degradation patterns.

**Split strategy (implemented in `src/split.py`):**

- **Primary axis: time.** Snapshots are ordered by calendar cycle; the final ~20% of the timeline (per the last snapshot date) forms the **test window**, an earlier ~10% band forms the **validation window**, and everything before is **training**. This mirrors deployment: we always train on the past and score the future.
- **Group disjointness on top of time:** a component instance (and by extension its aircraft) that has any snapshot in the test window is **excluded from training entirely**; likewise for validation. Group assignment is done by hashing `component_id` (stable across runs), so the split is deterministic and reproducible.
- **No future information in features.** All rolling/aggregate features are computed with a strict **as-of** semantics: a snapshot at cycle N may only use information from cycles ≤ N. The label itself is the only forward-looking field (30-cycle window), which is the target, not a feature.
- **No test-set touch during fitting.** Imputation statistics, scaling, categorical mappings, and threshold selection all happen on train/validation only; the test set is scored once, at the end.

---

## 3. Modeling

### 3.1 Feature engineering (`src/features.py`)

Beyond the raw snapshot fields, we build per-component history features computed strictly as-of the snapshot:

- **Sensor trend features:** rolling mean / std / delta over the last 5 and 15 checkpoints for each sensor — degradation is a *trend*, so a rising vibration baseline matters more than one high reading.
- **Ratio to fleet baseline:** current sensor value ÷ median value of the same `component_type` over the trailing window — normalizes across types without leaking future data.
- **Usage intensity:** cycles since install ÷ expected type lifetime (a crude utilization proxy).
- **Maintenance recency:** days since last shop visit, number of prior repairs (repeat offenders are riskier).
- **Fault-code dynamics:** counts in 30/90-day windows and a "hard fault in last 30 days" flag.

All aggregation windows end at the snapshot cycle; the generator and the feature builder share this as-of contract (enforced by a unit test).

### 3.2 Models compared

| Model | Why chosen |
|---|---|
| **Logistic Regression** (L2, class-balanced, RobustScaler) | Interpretable baseline; its coefficients give a first sanity check that engineered features point in physically sensible directions. |
| **Gradient-boosted trees — XGBoost** (`scale_pos_weight`, early stopping on validation PR-AUC) | Strong tabular baseline; captures non-linear thresholds and interactions (e.g. "high vibration *only matters when* oil temp is also rising"). Native NaN handling means raw sensor gaps don't require imputation. |

`src/train.py` runs both through the same split, the same features, and reports the same metric table, so the comparison is apples-to-apples. `HistGradientBoostingClassifier` is also run as a cross-check (native sklearn, no extra dependency).

### 3.3 Risk score

Raw model outputs are **post-hoc calibrated (Platt scaling fitted on the validation set)** — cost-sensitive training inflates probabilities, and calibration restores a meaningful scale without changing the ranking (monotonic transform). The planner-facing **risk score 0–100** is the percentile of the calibrated probability within the validation reference distribution, banded relative to the operating point:

| Band | Definition | Meaning |
|---|---|---|
| **High** | top 5% of the fleet | alert zone — matches the ≤5/100 budget |
| **Elevated** | next 10% | schedule inspection at next maintenance window |
| **Watch** | next 15% | trend monitoring |
| **Low** | remainder | routine |

Every alert ships with the top contributing factors per component (Section 5). Planners see the score, the band, and the *why* — not a raw model output.

---

## 4. Evaluation and Threshold Selection

### 4.1 Metrics

With ~2% positives, accuracy is meaningless. We report:

- **PR-AUC** (primary model-selection metric) — sensitivity to the minority class.
- **Recall at the operating point** — must be ≥ 80%.
- **Alert rate** — must be ≤ 5% (≤5 per 100).
- **ROC-AUC** and **Brier score** — secondary sanity checks.

**Operational semantics:** deployment scores every active component after each cycle data load, so **each snapshot row is one component-review**. The alert rate is therefore the fraction of component-reviews alerted in the window, and recall is the fraction of positive reviews (removals within 30 cycles) caught — both computed the same way in validation, test, and production.

### 4.2 Threshold selection procedure

Threshold is chosen **on the validation set only** (never test):

1. Sweep candidate thresholds over the validation score distribution.
2. Keep all thresholds where **alert rate ≤ 5%** on the per-component latest snapshots.
3. Among those, pick the threshold that **maximizes recall**.
4. Report the chosen threshold with its validation and (single, final) test performance.

The tension in the brief (≥80% recall at ≤5% alert rate) is a precision/recall trade-off that only a well-separated score distribution can satisfy; the report's results table shows whether each model clears it. The mock generator is tuned so the signal is realistic-but-learnable, i.e. the PoC demonstrates the *methodology*, not inflated numbers.

### 4.3 Cost framing

Every false alert costs a planning-hours and an unnecessary inspection; every miss costs an unscheduled removal (AOG risk, expedited logistics). The ≤5% alert-rate cap is treated as a hard budget, and recall is maximized within it — matching how an MRO reliability board would actually set the operating point.

---

## 5. Explainability

Two levels, both implemented in `src/explain.py`:

1. **Global:** feature importance (gain-based for XGBoost, permutation importance as cross-check) — answers "what does the model look at overall?" Expected drivers: vibration trend, hard faults, age ratio, repair count.
2. **Local (per component):** for XGBoost we use native **SHAP values** (`pred_contribs`), so every alert ships with a ranked list like:
   > `High alert — component CMP-0041 (APU), risk 87/100. Top factors: vibration trend +Δ over last 15 checkpoints (contribution 0.31), hard fault FC-7723 in last 30 days (0.24), cycles-since-install / expected-life ratio 1.18 (0.12).`

   This "why" list is what makes the score actionable for reliability engineers and builds trust before the model influences work orders.
3. **Sanity guardrail:** per-sample top factors are checked against a physically-plausible allowlist direction (e.g. rising vibration should push risk *up*); if the model ever leans on a nonsensical feature, that surfaces in review before deployment.

Example alert cards for real test-set components are emitted to `outputs/example_alerts.md`.

---

## 6. Production Design

### 6.1 Deployment

- **Batch scoring service.** After each flight-cycle data load (nightly), the pipeline ingests new snapshots, recomputes as-of features, and scores every active component. Scores and alert cards are written to the reliability dashboard and pushed to the **MRO planning queue** (work-order system integration via REST/MQ; planners accept/reject alerts, and those decisions are logged as feedback labels).
- **Serving contract.** Model artifact + preprocessing are versioned together (model registry, e.g. MLflow); the scoring job pins the artifact hash so training and serving transformations can never drift apart.
- **Human in the loop.** Alerts are recommendations, not auto-work-orders; a reliability engineer signs off. Accept/reject feedback becomes training signal for future iterations.

### 6.2 Monitoring

| Signal | Method | Response |
|---|---|---|
| Feature drift | PSI per feature vs. training reference | Investigate upstream data; may trigger retrain |
| Score drift | Score distribution shift (KS / PSI) | Review before alert budget is disturbed |
| Alert rate | Alerts per 100 active components per window | Breach of the ≤5% budget → re-tune threshold |
| Realized recall (lagged) | Removals in last 30 cycles that we failed to flag (labels arrive with removal events) | Primary health metric; sustained drop → retrain |
| Feedback loop | Planner accept/reject rate per band | Detects trust decay and label noise |

### 6.3 Retraining

- **Scheduled:** quarterly refresh (fleet composition and maintenance programs change slowly).
- **Triggered:** feature PSI breach, realized-recall drop below target, or a new component type entering the fleet.
- Process: retrain on all data through the latest closed window, re-run the full split/threshold procedure, compare champion vs. challenger on PR-AUC + operating point, then promote via the registry with a shadow-mode week before cutover.

### 6.4 Handling new components and missing data in production

- **New component instance (cold start):** no history → model relies on type-level baseline features (type one-hots, fleet-baseline sensor ratios) and age. Until 15 checkpoints of history exist, its alerts are labeled "low confidence" and routed with a wider band.
- **New component *type* (never seen):** one-hot vectors are all-zero + a `type_seen` flag defaults the model to usage/maintenance features; the type is flagged for review and the next retrain incorporates it.
- **Missing sensors:** missing-indicator features + native NaN handling mean a dead sensor *degrades gracefully* (score widens uncertainty rather than silently imputing optimism); a sensor dark for >30 days raises a data-quality alert to the ops team.

### 6.5 Model drift definition (for the brief)

We treat drift as a **monitored, trigger-based process**, not a scheduled guess: PSI on inputs, distribution shift on outputs, and lagged realized recall on outcomes. Any breach re-opens the threshold-tuning/retraining workflow in 6.3.

---

## 7. Limitations and Future Improvements

1. **Synthetic data.** The mock generator encodes plausible degradation physics, but real MRO data has messier failure modes (multi-cause removals, data-entry noise, inconsistent fault-code taxonomies). The pipeline is dataset-agnostic; real data mainly changes cleaning rules and thresholds.
2. **Label latency and censoring.** "Removed within 30 cycles" labels finalize only after the window closes; components still healthy at dataset end are right-censored. A survival-analysis formulation (e.g. discrete-time hazard models) would use time more efficiently and is the top modeling upgrade.
3. **No repair-effect modeling.** We know repairs happened, not what they fixed; shop-visit outcome data would let us model post-maintenance resets in the degradation curve.
4. **Calibration.** Threshold tuning optimizes the operating point, but planner-facing risk bands would benefit from explicit probability calibration (isotonic on validation) and uncertainty estimates.
5. **Fleet-level effects.** Aircraft utilization, route profile, and operator habits are currently only implicit; hierarchical features (per-tail baseline) would sharpen this.
6. **Beyond snapshots.** Raw sensor streams (higher-frequency QAR data) and full fault-code text could feed sequence models; the current checkpoint design is deliberately simple for a PoC.

---

## 8. Repository Map

```
mro-removal-prediction/
├── docs/REPORT.md          # this report
├── data/                   # mock dataset (CSV) + packaging zip
├── src/
│   ├── generate_data.py    # synthetic dataset generator (~2% positives)
│   ├── prepare.py          # cleaning, missing data, outliers
│   ├── features.py         # as-of feature engineering
│   ├── split.py            # temporal + group-disjoint split
│   ├── train.py            # model comparison, training
│   ├── evaluate.py         # metrics, threshold selection, alert budget
│   └── explain.py          # global + local (SHAP) explanations
├── tests/                  # pytest suite (leakage, as-of contract, budget)
├── outputs/                # metrics tables, example alert cards
└── README.md               # how to run
```

## 9. Results Summary

*(reproducible via `python src/run_all.py`; generated table in `outputs/metrics.md`)*

**Dataset:** 84,000 snapshots · 1,471 component instances · 40 aircraft · positive rate 1.76%.
**Splits:** train 29,949 rows (536 components) / validation 5,460 / test 5,880 — test = 14 held-out aircraft, final 20% of the timeline (cycles ≥ 4,800).

| Model | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Recall @ op point | Alert rate /100 | Meets brief? |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.788 | 0.814 | 0.997 | 100.0% | 2.77% | YES |
| **XGBoost (primary)** | 0.794 | **0.814** | **0.997** | **98.6%** | **2.87%** | **YES** |
| HistGradientBoosting (cross-check) | 0.733 | 0.757 | 0.996 | 97.3% | 2.86% | YES |

- **Operating point:** threshold selected on validation only (XGBoost calibrated threshold ≈ 0.0122, i.e. the alert zone ≈ top 3% of the fleet) — comfortably inside the ≤5/100 budget while catching 98.6% of removals on unseen aircraft in a future window.
- **Comparison takeaway:** all three models clear the brief on this dataset because the precursor signal is strong; XGBoost wins on validation PR-AUC and generalizes best, and — critically — it ships with native SHAP explanations for every alert, which is what makes it the primary production candidate. The linear baseline remains valuable as a fast sanity model and an interpretable fallback.
- **Global drivers (XGBoost gain):** `sensor_vibration_ratio_baseline`, `sensor_vibration_std5`, `sensor_vibration_delta15`, then oil-temperature trend features — physically consistent with accelerating degradation.
- **Example alert** (full cards in `outputs/example_alerts.md`):
  > `CMP-01409 (Fuel_Pump, AC-038) — risk 100/100 HIGH. Top factors: vibration 2.6× fleet baseline (+3.28), 5-checkpoint vibration volatility (+1.33), oil temp 1.29× baseline (+1.23).`
