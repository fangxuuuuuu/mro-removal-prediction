"""As-of feature engineering.

Every feature at snapshot cycle N may only use information from checkpoints
<= N (strict as-of semantics, enforced by tests). Fleet/type baseline
statistics are computed from the *training* window only and then applied to
validation/test, so no future or evaluation-set information leaks in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import EXPECTED_LIFE, SENSOR_COLS

# Raw numeric fields used as features directly
BASE_NUMERIC = [
    "snapshot_cycle",
    "cycles_since_install",
    "fault_code_count_90d",
    "has_hard_fault",
    "days_since_last_maint",
    "repairs_last_180d",
]
CATEGORICAL = ["component_type"]
GROUP_COLS = ["snapshot_id", "aircraft_id", "component_id", "snapshot_cycle"]
TARGET = "label_removed_30d"

# Engineered rolling features per sensor
ROLL_WINDOWS = [5, 15]


def compute_type_baselines(train_df: pd.DataFrame) -> dict[str, float]:
    """Median sensor value per component_type, fitted on the training window."""
    baselines: dict[str, float] = {}
    for col in SENSOR_COLS:
        med = train_df.groupby("component_type")[col].median()
        for ctype, v in med.items():
            baselines[f"{col}__{ctype}"] = float(v) if pd.notna(v) else np.nan
    return baselines


def add_features(df: pd.DataFrame, baselines: dict[str, float] | None = None) -> pd.DataFrame:
    """Build model feature matrix (plus group/target columns for bookkeeping)."""
    out = df.copy()
    out = out.sort_values(["component_id", "snapshot_cycle"]).reset_index(drop=True)
    g = out.groupby("component_id", sort=False)

    # --- rolling sensor trend features (as-of safe: trailing windows) ---
    for col in SENSOR_COLS:
        roll = g[col].rolling(ROLL_WINDOWS[0], min_periods=1)
        out[f"{col}_mean5"] = roll.mean().reset_index(level=0, drop=True)
        roll15 = g[col].rolling(ROLL_WINDOWS[1], min_periods=2)
        out[f"{col}_mean15"] = roll15.mean().reset_index(level=0, drop=True)
        out[f"{col}_std5"] = g[col].rolling(ROLL_WINDOWS[0], min_periods=2).std().reset_index(level=0, drop=True)
        prev = g[col].shift(1)
        out[f"{col}_delta_prev"] = out[col] - prev
        prev15 = g[col].shift(ROLL_WINDOWS[1])
        out[f"{col}_delta15"] = out[col] - prev15
        # missing indicator: a dark sensor is itself a signal
        out[f"{col}_missing"] = out[col].isna().astype(int)

    # --- ratio to fleet baseline (training-window medians) ---
    if baselines:
        for col in SENSOR_COLS:
            base = out["component_type"].map(
                {k.split("__", 1)[1]: v for k, v in baselines.items() if k.startswith(f"{col}__")}
            )
            out[f"{col}_ratio_baseline"] = out[col] / base.replace(0, np.nan)

    # --- utilization: age vs expected type life ---
    out["age_ratio"] = out["cycles_since_install"] / out["component_type"].map(EXPECTED_LIFE)

    # --- maintenance recency normalized ---
    out["days_since_last_maint_log1p"] = np.log1p(out["days_since_last_maint"].clip(lower=0))

    feature_cols = (
        BASE_NUMERIC
        + SENSOR_COLS
        + [c for c in out.columns if c.endswith(("_mean5", "_mean15", "_std5", "_delta_prev", "_delta15", "_missing", "_ratio_baseline"))]
        + ["age_ratio", "days_since_last_maint_log1p"]
        + CATEGORICAL
    )
    # de-duplicate preserving order
    seen = set()
    feature_cols = [c for c in feature_cols if not (c in seen or seen.add(c))]

    keep = list(dict.fromkeys(GROUP_COLS + feature_cols + [TARGET]))
    return out[keep], feature_cols
