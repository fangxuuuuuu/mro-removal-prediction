"""Model training: Logistic Regression baseline vs gradient-boosted trees.

Both models consume identical features and the identical split so the
comparison is apples-to-apples. Class imbalance (~2% positives) is handled
with cost-sensitive learning, not resampling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import SENSOR_COLS
from features import BASE_NUMERIC, CATEGORICAL

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # pragma: no cover
    XGBClassifier = None
    HAS_XGB = False


MODEL_NAMES = ["logistic_regression", "xgboost"] if HAS_XGB else ["logistic_regression"]


def _numeric_feature_cols(feature_cols: list[str]) -> list[str]:
    return [c for c in feature_cols if c not in CATEGORICAL]


def _make_lr_pipeline(feature_cols: list[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, RobustScaler

    num_cols = _numeric_feature_cols(feature_cols)
    # Sensors stay raw (NaN -> imputer); engineered indicators are passthrough
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", RobustScaler())]),
             [c for c in num_cols if c not in {f"{s}_missing" for s in SENSOR_COLS}]),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01), CATEGORICAL),
        ],
        remainder="passthrough",
    )
    clf = LogisticRegression(
        C=0.5, max_iter=2000, class_weight="balanced", solver="lbfgs",
    )
    return Pipeline([("pre", pre), ("clf", clf)]), num_cols


def _make_xgb(feature_cols: list[str], y_train: pd.Series):
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    return XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=2.0,
        scale_pos_weight=neg / max(pos, 1),
        eval_metric="aucpr",
        early_stopping_rounds=40,
        enable_categorical=True,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def _prepare_xgb_matrix(X: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """XGBoost consumes raw NaNs; categoricals as pandas category dtype."""
    M = X[feature_cols].copy()
    for c in CATEGORICAL:
        M[c] = M[c].astype("category")
    return M


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_cols: list[str],
) -> dict[str, dict]:
    """Train every available model; return {name: {'model': fitted, 'predict_proba': fn}}."""
    trained: dict[str, dict] = {}

    # --- Logistic Regression baseline ---
    lr_pipeline, num_cols = _make_lr_pipeline(feature_cols)
    lr_pipeline.fit(X_train[feature_cols], y_train)
    trained["logistic_regression"] = {
        "model": lr_pipeline,
        "predict_proba": lambda X: lr_pipeline.predict_proba(X[feature_cols])[:, 1],
    }

    # --- XGBoost ---
    if HAS_XGB:
        M_train = _prepare_xgb_matrix(X_train, feature_cols)
        M_val = _prepare_xgb_matrix(X_val, feature_cols)
        xgb = _make_xgb(feature_cols, y_train)
        xgb.fit(M_train, y_train, eval_set=[(M_val, y_val)], verbose=False)
        trained["xgboost"] = {
            "model": xgb,
            "predict_proba": lambda X: xgb.predict_proba(_prepare_xgb_matrix(X, feature_cols))[:, 1],
        }

    # --- HistGradientBoosting cross-check (sklearn-native, NaN aware) ---
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import OrdinalEncoder

        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xtr = X_train[feature_cols].copy()
        Xva = X_val[feature_cols].copy()
        cat_idx = [feature_cols.index(c) for c in CATEGORICAL]
        Xtr_cat = enc.fit_transform(Xtr[CATEGORICAL])
        Xva_cat = enc.transform(Xva[CATEGORICAL])
        for i, c in enumerate(CATEGORICAL):
            Xtr[c] = Xtr_cat[:, i]
            Xva[c] = Xva_cat[:, i]
        hgb = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=5,
            categorical_features=cat_idx, class_weight="balanced", random_state=42,
        )
        hgb.fit(Xtr, y_train)

        def _hgb_proba(X: pd.DataFrame) -> np.ndarray:
            Xh = X[feature_cols].copy()
            for i, c in enumerate(CATEGORICAL):
                Xh[c] = enc.transform(Xh[[c]])[:, i]
            return hgb.predict_proba(Xh)[:, 1]

        trained["hist_gradient_boosting"] = {
            "model": hgb,
            "predict_proba": _hgb_proba,
        }
    except Exception:
        # Cross-check model is optional; failure here must not break the run.
        pass

    return trained
