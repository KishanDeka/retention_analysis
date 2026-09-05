"""Churn-model definitions and their shared preprocessing pipelines."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    X: pd.DataFrame,
    *,
    use_pca: bool = False,
    pca_variance: float = 0.95,
    random_state: int = 42,
    ) -> ColumnTransformer:
    """Create preprocessing from the supplied model-feature columns."""

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    numeric_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    if use_pca and numeric:
        numeric_steps.append(
            (
                "pca",
                PCA(
                    n_components=pca_variance,
                    svd_solver="full",
                    random_state=random_state,
                ),
            )
        )

    transformers: list[tuple[str, object, list[str]]] = []
    if numeric:
        transformers.append(("num", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("X must contain at least one model-feature column")
    return ColumnTransformer(transformers)


def build_logistic_model(
    X: pd.DataFrame, random_state: int = 42
    ) -> Pipeline:
    """Build the interpretable logistic-regression baseline."""

    return Pipeline(
        [
            ("preprocessor", build_preprocessor(X, random_state=random_state)),
            (
                "model",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_logistic_pca_model(
    X: pd.DataFrame,
    random_state: int = 42,
    pca_variance: float = 0.95,
    ) -> Pipeline:
    """Build logistic regression with PCA on numeric predictors."""

    return Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(
                    X,
                    use_pca=True,
                    pca_variance=pca_variance,
                    random_state=random_state,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_random_forest_model(
    X: pd.DataFrame, random_state: int = 42
    ) -> Pipeline:
    """Build the nonlinear random-forest churn model."""

    return Pipeline(
        [
            ("preprocessor", build_preprocessor(X, random_state=random_state)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=350,
                    min_samples_leaf=6,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_gradient_boosting_model(
    X: pd.DataFrame, random_state: int = 42
    ) -> Pipeline:
    """Build the histogram gradient-boosting churn model."""

    return Pipeline(
        [
            ("preprocessor", build_preprocessor(X, random_state=random_state)),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=250,
                    learning_rate=0.06,
                    max_leaf_nodes=24,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_models(
        X: pd.DataFrame,
    random_state: int = 42,
    pca_variance: float = 0.95,
    ) -> dict[str, Pipeline]:
    return {
        "logistic_regression": build_logistic_model(X, random_state),
        "logistic_regression_pca": build_logistic_pca_model(
            X, random_state, pca_variance
        ),
        "random_forest": build_random_forest_model(X, random_state),
        "gradient_boosting": build_gradient_boosting_model(X, random_state),
        "xgboost": build_xgboost_model(X, random_state),
    }


def build_xgboost_model(
    X: pd.DataFrame, random_state: int = 42, **overrides
    ) -> Pipeline:
    """Build optional XGBoost using the same preprocessing contract."""

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("Install xgboost to use build_xgboost_model") from exc

    parameters = {
        "n_estimators": 250,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": random_state,
    }
    parameters.update(overrides)
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(X, random_state=random_state)),
            ("model", XGBClassifier(**parameters)),
        ]
    )


def build_lightgbm_model(
    X: pd.DataFrame, random_state: int = 42, **overrides
    ) -> Pipeline:
    """Build optional LightGBM using the same preprocessing contract."""

    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError("Install lightgbm to use build_lightgbm_model") from exc

    parameters = {
        "n_estimators": 250,
        "learning_rate": 0.05,
        "random_state": random_state,
    }
    parameters.update(overrides)
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(X, random_state=random_state)),
            ("model", LGBMClassifier(**parameters)),
        ]
    )
