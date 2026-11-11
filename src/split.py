"""Leakage-safe train/validation/test split.

Strategy (see docs/REPORT.md §2):
1. Hold out a set of *aircraft* entirely (stable hash) -> test fleet. This
   prevents aircraft-level leakage: no tail is seen in both training and test.
2. The test fleet is scored only on snapshots in the final `test_frac` of the
   timeline -> temporal separation (we always score "the future").
3. On the remaining (development) fleet, a temporal cut creates a validation
   band before the test window; training uses everything earlier.
4. Component-level leakage: any component instance alive at the validation
   boundary (i.e. with snapshots at/after val_start) is removed from training,
   because its snapshots are highly autocorrelated.
"""

from __future__ import annotations

import zlib

import pandas as pd

from config import SEED


def _aircraft_bucket(aircraft_id: str, n_buckets: int, salt: int = SEED) -> int:
    """Stable per-aircraft hash bucket (deterministic across runs)."""
    h = zlib.crc32(f"{salt}:{aircraft_id}".encode()) % n_buckets
    return h


def make_splits(
    df: pd.DataFrame,
    test_aircraft_frac: float = 0.30,
    test_frac: float = 0.20,
    val_frac: float = 0.10,
) -> dict[str, pd.DataFrame]:
    """Return dict with keys train / validation / test."""
    df = df.sort_values("snapshot_cycle").reset_index(drop=True)
    cycles = df["snapshot_cycle"].unique()
    n_cyc = len(cycles)

    test_start = cycles[int(n_cyc * (1.0 - test_frac))]
    val_start = cycles[int(n_cyc * (1.0 - test_frac - val_frac))]

    # 1) hold out whole aircraft for the test fleet
    n_buckets = max(int(round(1.0 / test_aircraft_frac)), 2)
    test_aircraft = {
        ac for ac in df["aircraft_id"].unique()
        if _aircraft_bucket(ac, n_buckets) == 0
    }

    is_test_fleet = df["aircraft_id"].isin(test_aircraft)
    dev = df[~is_test_fleet]

    # 2) temporal windows
    dev_train_band = dev[dev["snapshot_cycle"] < val_start]
    dev_val_band = dev[(dev["snapshot_cycle"] >= val_start) & (dev["snapshot_cycle"] < test_start)]

    # 3) component-level disjointness: parts alive at val_start cannot be in train
    val_components = set(dev_val_band["component_id"])
    train = dev_train_band[~dev_train_band["component_id"].isin(val_components)]

    # 4) test fleet, final window only (simulates scoring the future)
    test = df[is_test_fleet & (df["snapshot_cycle"] >= test_start)]

    return {
        "train": train.reset_index(drop=True),
        "validation": dev_val_band.reset_index(drop=True),
        "test": test.reset_index(drop=True),
        "test_aircraft": sorted(test_aircraft),
        "test_start_cycle": int(test_start),
        "val_start_cycle": int(val_start),
    }
