import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import LABEL_WINDOW, SENSOR_COLS
from evaluate import (
    ALERT_RATE_CAP,
    alert_rate,
    classification_metrics,
    latest_snapshot_mask,
    select_threshold,
)
from features import add_features, compute_type_baselines
from prepare import prepare
from split import make_splits


# ----------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def raw_df():
    from generate_data import generate_dataset
    return generate_dataset(n_aircraft=6, seed=7)


@pytest.fixture(scope="module")
def prepared_df(raw_df):
    return prepare(raw_df)


@pytest.fixture(scope="module")
def splits(prepared_df):
    return make_splits(prepared_df)


# ------------------------------------------------------------- data quality
def test_generator_produces_expected_columns(raw_df):
    expected = {
        "snapshot_id", "aircraft_id", "component_id", "component_type",
        "ata_chapter", "snapshot_cycle", "cycles_since_install",
        *SENSOR_COLS, "fault_code_count_90d", "has_hard_fault",
        "days_since_last_maint", "repairs_last_180d", "label_removed_30d",
    }
    assert expected.issubset(raw_df.columns)


def test_positive_rate_in_realistic_band(raw_df):
    """Mock brief says ~2% positives; tolerate 0.5%-5% for small fixtures."""
    rate = raw_df["label_removed_30d"].mean()
    assert 0.005 <= rate <= 0.05, f"positive rate {rate:.3%} outside band"


def test_labels_are_forward_looking(raw_df):
    """A positive label must correspond to an unscheduled removal that actually
    happens within LABEL_WINDOW cycles after the snapshot, and snapshots stop
    at removal."""
    for comp_id, grp in raw_df.groupby("component_id"):
        label_rows = grp[grp["label_removed_30d"] == 1]
        if len(grp) < 2:
            continue
        # all labels of one part occur in a single trailing cluster
        if len(label_rows):
            last_cycle = grp["snapshot_cycle"].max()
            first_label = label_rows["snapshot_cycle"].min()
            # removal happens at most LABEL_WINDOW cycles after the first labeled snapshot
            assert last_cycle - first_label <= LABEL_WINDOW


# ------------------------------------------------------------------ prepare
def test_prepare_clips_sensor_values(prepared_df):
    from config import SENSOR_LIMITS
    for col, (lo, hi) in SENSOR_LIMITS.items():
        assert prepared_df[col].min() >= lo
        assert prepared_df[col].max() <= hi


# -------------------------------------------------------------------- split
def test_split_disjoint_aircraft(splits):
    test_ac = set(splits["test"]["aircraft_id"])
    train_ac = set(splits["train"]["aircraft_id"])
    val_ac = set(splits["validation"]["aircraft_id"])
    assert not (test_ac & train_ac), "aircraft-level leakage: same tail in train and test"
    assert not (test_ac & val_ac), "aircraft-level leakage: same tail in validation and test"
    assert test_ac, "test fleet must not be empty"


def test_split_disjoint_components(splits):
    test_c = set(splits["test"]["component_id"])
    train_c = set(splits["train"]["component_id"])
    val_c = set(splits["validation"]["component_id"])
    assert not (test_c & train_c), "component-level leakage: same part in train and test"
    assert not (test_c & val_c), "component-level leakage: same part in validation and test"


def test_split_temporal_ordering(splits):
    assert splits["val_start_cycle"] < splits["test_start_cycle"]
    assert splits["train"]["snapshot_cycle"].max() < splits["test"]["snapshot_cycle"].min()
    assert splits["validation"]["snapshot_cycle"].min() >= splits["val_start_cycle"]
    assert splits["test"]["snapshot_cycle"].min() >= splits["test_start_cycle"]


# ----------------------------------------------------------------- features
def test_features_are_as_of_safe():
    """Changing a FUTURE checkpoint's sensor value must not change features of
    EARLIER checkpoints (strict as-of contract)."""
    rows = []
    for i, cycle in enumerate([0, 20, 40, 60]):
        rows.append(dict(
            snapshot_id=f"s{i}", aircraft_id="AC-1", component_id="CMP-1",
            component_type="APU", ata_chapter="49", snapshot_cycle=cycle,
            cycles_since_install=cycle, sensor_vibration=0.3 + 0.1 * i,
            sensor_oil_temp=400.0, sensor_pressure_delta=0.2,
            fault_code_count_90d=0, has_hard_fault=0,
            days_since_last_maint=10, repairs_last_180d=0, label_removed_30d=0,
        ))
    df_a = pd.DataFrame(rows)
    df_b = df_a.copy()
    # perturb the LAST checkpoint only
    df_b.loc[df_b.index[-1], "sensor_vibration"] = 99.0

    base = compute_type_baselines(df_a)
    fa, cols = add_features(df_a, base)
    fb, _ = add_features(df_b, base)

    early = fa["snapshot_cycle"] < 60
    pd.testing.assert_frame_equal(
        fa.loc[early].reset_index(drop=True),
        fb.loc[fb["snapshot_cycle"] < 60].reset_index(drop=True),
    )


def test_feature_matrix_has_no_raw_group_keys_as_features():
    from features import CATEGORICAL
    assert "aircraft_id" not in CATEGORICAL
    assert "component_id" not in CATEGORICAL


# ----------------------------------------------------------------- evaluate
def test_latest_snapshot_mask_one_row_per_component():
    df = pd.DataFrame({
        "component_id": ["A", "A", "B"],
        "snapshot_cycle": [10, 20, 5],
    })
    m = latest_snapshot_mask(df)
    assert m.sum() == 2
    assert sorted(df.loc[m, "snapshot_cycle"]) == [5, 20]


def test_select_threshold_respects_alert_budget():
    """Synthetic case: 200 components, 10 positives; scores make a clean cut."""
    n_comp, snaps = 200, 5
    rows = []
    scores = []
    labels = []
    for c in range(n_comp):
        for s in range(snaps):
            rows.append((f"C{c:03d}", s))
            if s == snaps - 1:  # operational view = latest snapshot
                if c < 10:
                    scores.append(0.95)  # 10 true positives at top
                    labels.append(1)
                elif c < 15:
                    scores.append(0.90)  # 5 false alarms
                    labels.append(0)
                else:
                    scores.append(0.05)
                    labels.append(0)
            else:
                scores.append(0.05)
                labels.append(0)
    df = pd.DataFrame(rows, columns=["component_id", "snapshot_cycle"])
    y = np.array(labels)
    sc = np.array(scores)

    op = select_threshold(y, sc, df)
    ar = alert_rate(sc, op["threshold"])
    assert ar <= ALERT_RATE_CAP
    # best available recall = 10/10 true positives with only 5 false alerts (alert rate 2.5%)
    assert op["recall"] == pytest.approx(1.0)


def test_classification_metrics_sanity():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.1, 0.2])
    m = classification_metrics(y, s, 0.5)
    assert m["recall"] == 1.0
    assert m["precision"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["tp"] == 2 and m["fp"] == 0 and m["fn"] == 0
