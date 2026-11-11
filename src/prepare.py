"""Data preparation: validation, dtype hygiene, physical-limit clipping.

Clipping (not dropping) preserves extreme readings — in this domain an
out-of-range sensor value is often exactly the precursor we want the model
to see. Values are clipped to the engineering limits in `config.py`.
"""

from __future__ import annotations

import pandas as pd

from config import SENSOR_COLS, SENSOR_LIMITS

INT_COLS = [
    "snapshot_cycle", "cycles_since_install", "fault_code_count_90d",
    "has_hard_fault", "days_since_last_maint", "repairs_last_180d",
    "label_removed_30d",
]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw snapshots dataframe. Returns a new dataframe."""
    out = df.copy()

    # dtypes
    for col in INT_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    for col in SENSOR_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # physical-limit clipping for sensors
    for col, (lo, hi) in SENSOR_LIMITS.items():
        if col in out.columns:
            out[col] = out[col].clip(lower=lo, upper=hi)

    # categorical hygiene
    out["component_type"] = out["component_type"].astype(str).str.strip()
    out["ata_chapter"] = out["ata_chapter"].astype(str).str.strip()

    # group keys must be strings without whitespace issues
    out["aircraft_id"] = out["aircraft_id"].astype(str).str.strip()
    out["component_id"] = out["component_id"].astype(str).str.strip()

    return out
