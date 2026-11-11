"""Explainability: global importance + per-alert local explanations.

For XGBoost we use native SHAP values via `pred_contribs` (TreeSHAP) — no
extra dependency required. For the logistic baseline, global coefficients and
coefficient-times-value local contributions are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def global_importance_xgb(xgb_model) -> pd.DataFrame:
    booster = xgb_model.get_booster()
    gain = booster.get_score(importance_type="gain")
    if not gain:
        return pd.DataFrame(columns=["feature", "gain"])
    imp = pd.DataFrame({"feature": list(gain.keys()), "gain": list(gain.values())})
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
    return imp


def local_contributions_xgb(xgb_model, X: pd.DataFrame, feature_cols: list[str]):
    """Return (n_samples, n_features) SHAP contribution matrix via TreeSHAP."""
    import xgboost as xgb

    M = X[feature_cols].copy()
    for c in M.columns:
        if str(M[c].dtype) == "object" or str(M[c].dtype) == "category":
            M[c] = M[c].astype("category")
    dmat = xgb.DMatrix(M, enable_categorical=True)
    contribs = xgb_model.get_booster().predict(dmat, pred_contribs=True)
    return np.asarray(contribs)[:, :-1]  # drop bias column


def local_contributions_lr(lr_pipeline, X: pd.DataFrame, feature_cols: list[str]):
    """Local contributions for the linear model: coef * (transformed value)."""
    pre = lr_pipeline.named_steps["pre"]
    clf = lr_pipeline.named_steps["clf"]
    Xt = pre.transform(X[feature_cols])
    coefs = clf.coef_[0]
    # Column order: num block (scaled), then one-hot block, then passthrough.
    n_num = len([c for c in feature_cols if c != "component_type"
                 and not c.endswith("_missing")])
    # Build a sparse-friendly dense contribution: use only for reporting top-k
    contribs = np.asarray(Xt.todense() if hasattr(Xt, "todense") else Xt) * coefs
    names = pre.get_feature_names_out()
    return contribs, list(names)


def explain_alert(shap_row: np.ndarray, feature_names: list[str],
                  row: pd.Series, top_k: int = 4) -> str:
    """Render one human-readable alert card from local contributions."""
    order = np.argsort(-np.abs(shap_row))[:top_k]
    factors = []
    for i in order:
        name = feature_names[i]
        contrib = shap_row[i]
        value = row.get(name, "")
        if isinstance(value, float):
            value = f"{value:.3f}"
        if isinstance(value, str) and value == "nan" or (not isinstance(value, str) and pd.isna(value)):
            value = "n/a"
        factors.append(f"{name}={value} (contribution {contrib:+.2f})")
    return "; ".join(factors)


def example_alert_cards(
    test_df: pd.DataFrame,
    test_percentile_scores: np.ndarray,
    contribs: np.ndarray,
    feature_names: list[str],
    threshold_pct: float,
    top_n: int = 10,
) -> str:
    """Markdown cards for the highest-risk components in the test window.

    `test_percentile_scores` are 0-100 fleet-percentile risk scores;
    `threshold_pct` is the operating point expressed in the same scale.
    """
    df = test_df.copy()
    df["risk_score"] = test_percentile_scores
    idx = df.groupby("component_id")["snapshot_cycle"].idxmax()
    latest = df.loc[idx].sort_values("risk_score", ascending=False)
    latest = latest[latest["risk_score"] >= threshold_pct][:top_n]

    cards = []
    for pos, (_, row) in enumerate(latest.iterrows(), start=1):
        i = df.index.get_loc(row.name)
        band = "HIGH" if row["risk_score"] >= 95 else "ELEVATED"
        card = (
            f"### Alert {pos} — {row['component_id']} ({row['component_type']}, "
            f"aircraft {row['aircraft_id']})\n"
            f"- **Risk score:** {row['risk_score']:.0f}/100 — {band}\n"
            f"- **Snapshot cycle:** {int(row['snapshot_cycle'])} · "
            f"age {int(row['cycles_since_install'])} cycles · "
            f"hard fault: {'yes' if row['has_hard_fault'] else 'no'}\n"
            f"- **Top factors:** {explain_alert(contribs[i], feature_names, row)}\n"
        )
        cards.append(card)
    return "\n".join(cards) if cards else "_No components above threshold in the test window._"
