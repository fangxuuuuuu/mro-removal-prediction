"""Evaluation, operating-point (threshold) selection, and alert-budget math.

Operational semantics (mirrors deployment): after every cycle data load the
pipeline scores every active component — so **each snapshot row is one
component-review**. The alert budget "≤5 alerts per 100 active components per
review window" therefore translates to: fraction of component-reviews alerted
must stay ≤ 5%, and recall = fraction of positive reviews (removals within 30
cycles) that are alerted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

ALERT_RATE_CAP = 0.05  # 5 alerts / 100 active component-reviews


def latest_snapshot_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask selecting the latest snapshot per component (diagnostic view)."""
    idx = df.groupby("component_id")["snapshot_cycle"].idxmax()
    mask = pd.Series(False, index=df.index)
    mask.loc[idx] = True
    return mask


def alert_rate(scores: np.ndarray, threshold: float) -> float:
    """Fraction of component-reviews alerted (alerts per 100 active components)."""
    return float((scores >= threshold).mean())


def select_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    df: pd.DataFrame | None = None,
    alert_rate_cap: float = ALERT_RATE_CAP,
) -> dict:
    """Pick the operating point on the validation set.

    Procedure (REPORT.md §4.2):
      1. sweep candidate thresholds over the validation score distribution,
      2. keep those meeting the alert-rate budget,
      3. among them maximize recall (tie-break: lower alert rate).
    `df` is accepted for API compatibility and unused in the row-review protocol.
    """
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 401)))
    best = None
    n_pos = int((y_true == 1).sum())
    for thr in candidates:
        ar = float((scores >= thr).mean())
        if ar > alert_rate_cap:
            continue
        tp = int(((scores >= thr) & (y_true == 1)).sum())
        recall = tp / n_pos if n_pos else 0.0
        if best is None or recall > best["recall"] or (
            recall == best["recall"] and ar < best["alert_rate"]
        ):
            best = {"threshold": float(thr), "recall": recall, "alert_rate": ar}
    if best is None:  # budget unsatisfiable -> lowest-alert-threshold fallback
        thr = float(np.quantile(scores, 1.0 - alert_rate_cap))
        tp = int(((scores >= thr) & (y_true == 1)).sum())
        best = {
            "threshold": thr,
            "recall": tp / n_pos if n_pos else 0.0,
            "alert_rate": float((scores >= thr).mean()),
            "budget_unsatisfiable": True,
        }
    return best


def classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
        "precision": precision,
        "recall": recall,
        "tp": tp, "fp": fp, "fn": fn,
    }


def evaluate_model(name: str, y_val: np.ndarray, val_scores: np.ndarray,
                   val_df: pd.DataFrame, y_test: np.ndarray | None = None,
                   test_scores: np.ndarray | None = None,
                   test_df: pd.DataFrame | None = None) -> dict:
    """Full evaluation for one model: threshold on validation, metrics on both."""
    op = select_threshold(y_val, val_scores)
    val_metrics = classification_metrics(y_val, val_scores, op["threshold"])
    result = {
        "model": name,
        "threshold": op["threshold"],
        "val": {**val_metrics, "alert_rate": op["alert_rate"]},
    }
    if y_test is not None and test_scores is not None:
        test_metrics = classification_metrics(y_test, test_scores, op["threshold"])
        test_metrics["alert_rate"] = alert_rate(test_scores, op["threshold"])
        result["test"] = test_metrics
    return result


def risk_bands(percentile_scores: np.ndarray) -> list[str]:
    """Map percentile risk scores (0-100) to bands.

    Bands are defined relative to the operating point, not absolute probability:
    High = top 5% of the fleet (the alert zone, matching the ≤5/100 budget),
    Elevated = next 10%, Watch = next 15%, Low = the rest.
    """
    bands = []
    for s in percentile_scores:
        if s >= 95:
            bands.append("High")
        elif s >= 85:
            bands.append("Elevated")
        elif s >= 70:
            bands.append("Watch")
        else:
            bands.append("Low")
    return bands


def score_to_percentile(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Percentile (0-100) of each score within a reference distribution
    (the validation-set calibrated scores). Monotonic, so rankings are kept."""
    ref = np.sort(reference)
    idx = np.searchsorted(ref, scores, side="left")
    return idx / len(ref) * 100.0
