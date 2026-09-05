"""PCA validation and supervised churn-model comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

try:
    from .models import build_models
except ImportError:  # Allows direct execution when src is on PYTHONPATH.
    from models import build_models


MODEL_LEAKAGE_COLUMNS = {
    "customer_id",
    "churn_flag",
    "churn_value",
    "churn_label",
    "customer_status",
    "churn_reason",
    "churn_category",
    "churn_score",
}


@dataclass
class ChurnAnalysis:
    model: Pipeline
    leaderboard: pd.DataFrame
    pca_validation: pd.DataFrame
    predictions: pd.DataFrame
    importance: pd.DataFrame
    X_test: pd.DataFrame


def add_business_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable reporting segments to the SQL-cleaned Telco data.

    ``customer_segment`` and ``monthly_revenue_at_risk`` are reporting fields.
    They are deliberately removed by :func:`prepare_churn_xy` before training.
    """

    df = frame.copy()
    tenure = df.get("tenure_in_months", pd.Series(np.nan, index=df.index))
    monthly_charge = df.get(
        "monthly_charge", pd.Series(np.nan, index=df.index)
    )

    df["tenure_band"] = pd.cut(
        tenure,
        bins=[-np.inf, 6, 12, 24, 48, np.inf],
        labels=[
            "0-6 months",
            "7-12 months",
            "13-24 months",
            "25-48 months",
            "49+ months",
        ],
    )

    contract = df.get("contract", pd.Series("", index=df.index)).astype("string")
    contract = contract.fillna("").str.strip().str.lower()
    support_column = (
        "premium_tech_support"
        if "premium_tech_support" in df.columns
        else "tech_support"
    )
    support = df.get(support_column, pd.Series("", index=df.index)).astype("string")
    support = support.fillna("").str.strip().str.lower()

    median_charge = monthly_charge.median()
    new_customer = tenure.le(6)
    month_to_month = contract.eq("month-to-month")
    high_value = monthly_charge.ge(median_charge)
    no_support = support.ne("yes")

    df["customer_segment"] = np.select(
        [
            new_customer & month_to_month,
            high_value & month_to_month,
            month_to_month & no_support,
            high_value,
        ],
        [
            "new_and_fragile",
            "high_value_at_risk",
            "unsupported_flexible",
            "high_value_loyal",
        ],
        default="core_customers",
    )

    if "churn_value" in df.columns:
        df["monthly_revenue_at_risk"] = monthly_charge * df["churn_value"]
    else:
        df["monthly_revenue_at_risk"] = np.nan
    return df


def prepare_churn_xy(
    frame: pd.DataFrame, target: str = "churn_value"
    ) -> tuple[pd.DataFrame, pd.Series]:
    """Select model inputs from an already cleaned SQL result."""

    if target not in frame:
        raise ValueError(f"Missing churn target column: {target}")
    usable = frame.loc[frame[target].notna()].copy()
    if usable.empty:
        raise ValueError("No rows have a non-null churn target")
    y = usable[target].astype(int)
    if not set(y.unique()).issubset({0, 1}) or y.nunique() < 2:
        raise ValueError("The churn target must contain both binary classes 0 and 1")

    excluded = MODEL_LEAKAGE_COLUMNS.union({target})
    X = usable.drop(columns=list(excluded), errors="ignore")
    X = X.drop(
        columns=["customer_segment", "monthly_revenue_at_risk"], errors="ignore"
    )
    if X.shape[1] == 0:
        raise ValueError("No predictor columns remain after leakage removal")
    return X, y


def compare_pca_validation(leaderboard: pd.DataFrame) -> pd.DataFrame:
    """Report whether PCA measurably improves the matched logistic baseline."""

    indexed = leaderboard.set_index("model")
    names = ["logistic_regression", "logistic_regression_pca"]
    if any(name not in indexed.index for name in names):
        raise ValueError("Leaderboard must include logistic models with and without PCA")
    rows = []
    for metric, higher_is_better in {
        "cv_roc_auc": True,
        "cv_pr_auc": True,
        "cv_brier_score": False,
    }.items():
        baseline = float(indexed.loc[names[0], metric])
        pca = float(indexed.loc[names[1], metric])
        raw_delta = pca - baseline
        benefit = raw_delta if higher_is_better else -raw_delta
        rows.append(
            {
                "metric": metric,
                "without_pca": baseline,
                "with_pca": pca,
                "pca_delta": raw_delta,
                "pca_benefit": benefit,
                "pca_improved": benefit > 0,
            }
        )
    return pd.DataFrame(rows)


def run_churn_analysis(
    frame: pd.DataFrame,
    random_state: int = 42,
    pca_variance: float = 0.95,
    ) -> ChurnAnalysis:
    X, y = prepare_churn_xy(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "neg_brier": "neg_brier_score",
    }
    models = build_models(X_train, random_state, pca_variance)
    rows: list[dict[str, float | str]] = []
    for name, pipeline in models.items():
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
        )
        rows.append(
            {
                "model": name,
                "cv_roc_auc": scores["test_roc_auc"].mean(),
                "cv_roc_auc_std": scores["test_roc_auc"].std(),
                "cv_pr_auc": scores["test_pr_auc"].mean(),
                "cv_pr_auc_std": scores["test_pr_auc"].std(),
                "cv_brier_score": -scores["test_neg_brier"].mean(),
            }
        )
    leaderboard = (
        pd.DataFrame(rows)
        .sort_values(["cv_pr_auc", "cv_roc_auc"], ascending=False)
        .reset_index(drop=True)
    )
    pca_validation = compare_pca_validation(leaderboard)

    best_name = str(leaderboard.loc[0, "model"])
    best = models[best_name].fit(X_train, y_train)
    probability = best.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    holdout = {
        "holdout_roc_auc": roc_auc_score(y_test, probability),
        "holdout_pr_auc": average_precision_score(y_test, probability),
        "holdout_brier_score": brier_score_loss(y_test, probability),
        "holdout_precision": precision_score(y_test, prediction, zero_division=0),
        "holdout_recall": recall_score(y_test, prediction, zero_division=0),
        "holdout_f1": f1_score(y_test, prediction, zero_division=0),
    }
    for metric, value in holdout.items():
        leaderboard.loc[leaderboard["model"].eq(best_name), metric] = value

    report_columns = [
        c
        for c in ["customer_id", "customer_segment", "monthly_charge", "cltv"]
        if c in frame
    ]
    predictions = frame.loc[X_test.index, report_columns].copy()
    predictions["actual_churn"] = y_test
    predictions["churn_probability"] = probability
    fallback_value = pd.Series(1.0, index=predictions.index)
    value = predictions.get(
        "cltv", predictions.get("monthly_charge", fallback_value)
    )
    predictions["value_at_risk"] = predictions["churn_probability"] * pd.to_numeric(
        value, errors="coerce"
    ).fillna(0)
    predictions = predictions.sort_values("value_at_risk", ascending=False)

    perm = permutation_importance(
        best,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=8,
        random_state=random_state,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {"feature": X_test.columns, "permutation_importance": perm.importances_mean}
    ).sort_values("permutation_importance", ascending=False)
    return ChurnAnalysis(
        best, leaderboard, pca_validation, predictions, importance, X_test
    )


def shap_importance(
    analysis: ChurnAnalysis, max_rows: int = 500, random_state: int = 42
    ) -> pd.DataFrame:
    """Calculate global SHAP importance for the selected fitted pipeline."""

    explanation, _ = compute_shap_explanation(
        analysis, max_rows=max_rows, random_state=random_state
    )
    return pd.DataFrame(
        {
            "feature": explanation.feature_names,
            "mean_abs_shap": np.abs(explanation.values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)


def compute_shap_explanation(
    analysis: ChurnAnalysis,
    rows=None,
    max_rows: int = 500,
    random_state: int = 42,
    ):
    """Return a SHAP Explanation and its transformed feature frame.

    Pass ``rows`` as customer indices for customer-specific waterfall plots.
    When omitted, a reproducible sample of the holdout data is explained.
    """

    import shap

    if rows is None:
        X = analysis.X_test.sample(
            min(max_rows, len(analysis.X_test)), random_state=random_state
        )
    else:
        X = analysis.X_test.loc[rows].head(max_rows)
    preprocessor = analysis.model.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()
    transformed_frame = pd.DataFrame(
        transformed, index=X.index, columns=feature_names
    )
    estimator = analysis.model.named_steps["model"]
    explainer = shap.Explainer(estimator, transformed_frame)
    explanation = explainer(transformed_frame)
    if explanation.values.ndim == 3:
        base_values = explanation.base_values
        if np.asarray(base_values).ndim == 2:
            base_values = np.asarray(base_values)[:, -1]
        explanation = shap.Explanation(
            values=explanation.values[:, :, -1],
            base_values=base_values,
            data=transformed_frame.to_numpy(),
            feature_names=list(feature_names),
        )
    elif explanation.feature_names is None:
        explanation.feature_names = list(feature_names)
    return explanation, transformed_frame
