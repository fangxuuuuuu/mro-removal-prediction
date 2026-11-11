"""End-to-end pipeline: generate -> prepare -> features -> split -> train
-> threshold/evaluate -> explain -> write outputs.

Run:  python src/run_all.py [--data data/component_snapshots.csv] [--no-generate]
Outputs: outputs/metrics.md, outputs/example_alerts.md, outputs/test_predictions.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import (
    evaluate_model,
    latest_snapshot_mask,
    risk_bands,
    score_to_percentile,
)
from explain import example_alert_cards, global_importance_xgb, local_contributions_xgb
from features import add_features, compute_type_baselines
from generate_data import generate_dataset
from prepare import prepare
from split import make_splits
from train import train_models

TARGET = "label_removed_30d"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/component_snapshots.csv")
    ap.add_argument("--no-generate", action="store_true",
                    help="reuse existing dataset CSV instead of regenerating")
    args = ap.parse_args()

    # ------------------------------------------------------------------ data
    if args.no_generate and os.path.exists(args.data):
        raw = pd.read_csv(args.data)
        print(f"loaded {len(raw):,} rows from {args.data}")
    else:
        raw = generate_dataset()
        os.makedirs(os.path.dirname(args.data), exist_ok=True)
        raw.to_csv(args.data, index=False)
        print(f"generated dataset -> {args.data}")

    df = prepare(raw)
    rate = df[TARGET].mean()
    print(f"positive rate: {rate:.3%}  rows={len(df):,}")

    # ----------------------------------------------------------------- split
    splits = make_splits(df)
    for k in ("train", "validation", "test"):
        s = splits[k]
        print(f"{k:>10}: {len(s):>7,} rows, {s['component_id'].nunique():>5} components, "
              f"positives={s[TARGET].sum():>5} ({s[TARGET].mean():.3%})")

    # --------------------------------------------------------------- features
    baselines = compute_type_baselines(splits["train"])
    train_df, feat_cols = add_features(splits["train"], baselines)
    val_df, _ = add_features(splits["validation"], baselines)
    test_df, _ = add_features(splits["test"], baselines)

    X_train, y_train = train_df, train_df[TARGET]
    X_val, y_val = val_df, val_df[TARGET]
    X_test, y_test = test_df, test_df[TARGET]

    # ----------------------------------------------------------------- train
    trained = train_models(X_train, y_train, X_val, y_val, feat_cols)
    print(f"trained models: {list(trained)}")

    # ------------------------------------------------------- calibration
    # Cost-sensitive training inflates raw probabilities (scale_pos_weight /
    # class_weight). Post-hoc Platt scaling on the validation set restores a
    # meaningful 0-100 risk score without changing the ranking (monotonic).
    from sklearn.linear_model import LogisticRegression as _LR

    val_scores_by_model: dict[str, np.ndarray] = {}
    test_scores_by_model: dict[str, np.ndarray] = {}

    for name, entry in trained.items():
        raw_val = entry["predict_proba"](X_val)
        raw_test = entry["predict_proba"](X_test)
        platt = _LR().fit(raw_val.reshape(-1, 1), y_val.to_numpy())

        def _calibrated(s: np.ndarray, _platt=platt) -> np.ndarray:
            return _platt.predict_proba(s.reshape(-1, 1))[:, 1]

        val_scores_by_model[name] = _calibrated(raw_val)
        test_scores_by_model[name] = _calibrated(raw_test)
        entry["predict_proba"] = _calibrated  # downstream users get calibrated scores

    # ------------------------------------------------------------- threshold
    results = []
    for name, entry in trained.items():
        val_scores = val_scores_by_model[name]
        test_scores = test_scores_by_model[name]
        res = evaluate_model(name, y_val.to_numpy(), val_scores, val_df,
                             y_test.to_numpy(), test_scores, test_df)
        results.append(res)
        t = res["test"]
        print(f"{name:>24}: val PR-AUC={res['val']['pr_auc']:.3f}  "
              f"test PR-AUC={t['pr_auc']:.3f}  recall={t['recall']:.1%}  "
              f"alert_rate={t['alert_rate']:.2%}  (thr={res['threshold']:.4f})")

    # ------------------------------------------------------------- explain
    primary = "xgboost" if "xgboost" in trained else next(iter(trained))
    entry = trained[primary]
    test_scores = test_scores_by_model[primary]

    os.makedirs("outputs", exist_ok=True)

    # metrics table
    lines = [
        "# Results Summary",
        "",
        f"Dataset: {len(df):,} snapshots · {df['component_id'].nunique():,} components · "
        f"{df['aircraft_id'].nunique()} aircraft · positive rate {rate:.2%}",
        "",
        f"Splits: train {len(splits['train']):,} / validation {len(splits['validation']):,} / "
        f"test {len(splits['test']):,} rows (test fleet: {len(splits['test_aircraft'])} aircraft, "
        f"cycles >= {splits['test_start_cycle']})",
        "",
        "| Model | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Recall @ op | Alert rate /100 | Meets brief? |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        t = r["test"]
        meets = "YES" if (t["recall"] >= 0.80 and t["alert_rate"] <= 0.05) else "NO"
        lines.append(
            f"| {r['model']} | {r['val']['pr_auc']:.3f} | {t['pr_auc']:.3f} | {t['roc_auc']:.3f} "
            f"| {t['recall']:.1%} | {t['alert_rate']:.2%} | {meets} |"
        )
    primary_thr = [r for r in results if r["model"] == primary][0]["threshold"]
    lines.append(f"\nOperating point (threshold) selected on validation only: "
                 f"{primary} thr={primary_thr:.4f}")
    with open("outputs/metrics.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    primary_res = [r for r in results if r["model"] == primary][0]
    primary_thr = primary_res["threshold"]

    # Planner-facing risk score: percentile of the calibrated probability
    # within the validation reference distribution (0-100, uniform spread).
    val_ref = val_scores_by_model[primary]
    test_pct = score_to_percentile(test_scores, val_ref)
    thr_pct = float((val_ref < primary_thr).mean() * 100)

    # predictions + bands
    pred_df = test_df[["snapshot_id", "aircraft_id", "component_id", "component_type",
                       "snapshot_cycle", TARGET]].copy()
    pred_df["risk_score"] = test_pct.round(1)
    pred_df["risk_band"] = risk_bands(test_pct)
    pred_df["alert"] = (test_scores >= primary_thr).astype(int)
    pred_df.to_csv("outputs/test_predictions.csv", index=False)

    # alert cards (XGBoost SHAP)
    cards_md = ["# Example Alert Cards (test window, highest-risk components)", ""]
    if primary == "xgboost":
        contribs = local_contributions_xgb(entry["model"], test_df, feat_cols)
        cards_md.append(example_alert_cards(
            test_df, test_pct, contribs, feat_cols, thr_pct,
        ))
        imp = global_importance_xgb(entry["model"])
        cards_md += ["", "# Global Feature Importance (XGBoost, gain)", ""]
        cards_md.append("| Rank | Feature | Gain |")
        cards_md.append("|---|---|---|")
        for i, row in imp.head(15).iterrows():
            cards_md.append(f"| {i + 1} | {row['feature']} | {row['gain']:.1f} |")
    else:
        cards_md.append("_XGBoost unavailable; cards require it._")
    with open("outputs/example_alerts.md", "w", encoding="utf-8") as f:
        f.write("\n".join(cards_md) + "\n")

    print("\nwrote outputs/metrics.md, outputs/example_alerts.md, outputs/test_predictions.csv")


if __name__ == "__main__":
    main()
