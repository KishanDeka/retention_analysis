from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from .features import make_preprocessor

@dataclass
class ModelResult:
    model: Any
    metrics: Dict[str, float]

def build_logistic_model(df, target="churn_flag"):
    preprocessor, _, _ = make_preprocessor(df, target=target)
    return Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=2000))
    ])

def evaluate_probabilistic_classifier(model, X_test, y_test):
    p = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": roc_auc_score(y_test, p),
        "pr_auc": average_precision_score(y_test, p),
        "brier_score": brier_score_loss(y_test, p),
    }

def fit_xgboost(df, target="churn_flag", **kwargs):
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise ImportError("Install xgboost to use fit_xgboost") from e
    preprocessor, _, _ = make_preprocessor(df, target=target)
    params = dict(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=42
    )
    params.update(kwargs)
    return Pipeline([("prep", preprocessor), ("model", XGBClassifier(**params))])

def fit_lightgbm(df, target="churn_flag", **kwargs):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as e:
        raise ImportError("Install lightgbm to use fit_lightgbm") from e
    preprocessor, _, _ = make_preprocessor(df, target=target)
    params = dict(n_estimators=250, learning_rate=0.05, random_state=42)
    params.update(kwargs)
    return Pipeline([("prep", preprocessor), ("model", LGBMClassifier(**params))])
