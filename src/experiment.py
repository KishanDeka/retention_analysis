"""Randomized A/B testing, heterogeneous uplift, and campaign decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import norm
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class BayesianABResult:
    summary: pd.DataFrame
    churn_reduction_samples: np.ndarray


@dataclass
class ExperimentAnalysis:
    ab_summary: pd.DataFrame
    bayesian_summary: pd.DataFrame
    scored_customers: pd.DataFrame
    uplift_deciles: pd.DataFrame
    qini_curve: pd.DataFrame
    policy_comparison: pd.DataFrame


def _validate_experiment(frame: pd.DataFrame) -> None:
    missing = sorted({"t", "y"}.difference(frame.columns))
    if missing:
        raise ValueError(f"Experiment data is missing required columns: {missing}")
    for column in ("t", "y"):
        values = set(frame[column].dropna().astype(int).unique())
        if not values.issubset({0, 1}) or values != {0, 1}:
            raise ValueError(f"{column} must contain both binary values 0 and 1")


def estimate_ab_effect(frame: pd.DataFrame) -> pd.DataFrame:
    """Estimate control-minus-treatment churn reduction with a z-test."""

    _validate_experiment(frame)
    control = frame.loc[frame["t"].eq(0), "y"].astype(float)
    treated = frame.loc[frame["t"].eq(1), "y"].astype(float)
    p_c, p_t = control.mean(), treated.mean()
    effect = p_c - p_t
    se_ci = np.sqrt(
        p_c * (1 - p_c) / len(control) + p_t * (1 - p_t) / len(treated)
    )
    pooled = (control.sum() + treated.sum()) / (len(control) + len(treated))
    se_null = np.sqrt(
        pooled * (1 - pooled) * (1 / len(control) + 1 / len(treated))
    )
    z = effect / se_null if se_null else np.nan
    p_value = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
    return pd.DataFrame(
        [
            {
                "control_customers": len(control),
                "treated_customers": len(treated),
                "control_churn_rate": p_c,
                "treated_churn_rate": p_t,
                "absolute_churn_reduction": effect,
                "relative_churn_reduction": effect / p_c if p_c else np.nan,
                "ci_95_low": effect - 1.96 * se_ci,
                "ci_95_high": effect + 1.96 * se_ci,
                "z_statistic": z,
                "p_value_two_sided": p_value,
                "number_needed_to_treat": 1 / effect if effect > 0 else np.nan,
            }
        ]
    )


def beta_posterior(
    successes: int,
    failures: int,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> tuple[float, float]:
    """Return posterior Beta parameters for a binary rate."""

    if successes < 0 or failures < 0:
        raise ValueError("successes and failures must be non-negative")
    return alpha_prior + successes, beta_prior + failures


def bayesian_ab_churn(
    frame: pd.DataFrame,
    *,
    draws: int = 100_000,
    seed: int = 42,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> BayesianABResult:
    """Estimate the posterior probability that treatment reduces churn."""

    _validate_experiment(frame)
    rng = np.random.default_rng(seed)
    control = frame.loc[frame["t"].eq(0), "y"].astype(int)
    treated = frame.loc[frame["t"].eq(1), "y"].astype(int)
    a_c, b_c = beta_posterior(
        int(control.sum()), int(len(control) - control.sum()), alpha_prior, beta_prior
    )
    a_t, b_t = beta_posterior(
        int(treated.sum()), int(len(treated) - treated.sum()), alpha_prior, beta_prior
    )
    control_samples = rng.beta(a_c, b_c, draws)
    treatment_samples = rng.beta(a_t, b_t, draws)
    reduction = control_samples - treatment_samples
    low, high = np.quantile(reduction, [0.025, 0.975])
    summary = pd.DataFrame(
        [
            {
                "posterior_control_churn": control_samples.mean(),
                "posterior_treated_churn": treatment_samples.mean(),
                "expected_churn_reduction": reduction.mean(),
                "credible_interval_95_low": low,
                "credible_interval_95_high": high,
                "probability_treatment_better": (reduction > 0).mean(),
            }
        ]
    )
    return BayesianABResult(summary, reduction)


def _campaign_pipeline(X: pd.DataFrame, random_state: int) -> Pipeline:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    transformers: list[tuple[str, object, list[str]]] = []
    if numeric:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    prep = ColumnTransformer(transformers)
    model = RandomForestClassifier(
        n_estimators=350,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline([("preprocessor", prep), ("model", model)])


def t_learner_uplift(
    X,
    treatment,
    outcome,
    base_model=None,
):
    """Fit a T-learner to numeric/model-ready features.

    This lower-level helper preserves the useful interface from ``causal.py``.
    For the raw Orange SQL table, prefer :func:`fit_t_learner`, which also
    handles mixed column types and creates an honest train/holdout split.
    """

    treatment_values = np.asarray(treatment)
    outcome_values = np.asarray(outcome).astype(int)
    if set(np.unique(treatment_values)) != {0, 1}:
        raise ValueError("treatment must contain both binary values 0 and 1")
    if base_model is None:
        base_model = RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
    model_control = clone(base_model)
    model_treated = clone(base_model)
    control_mask = treatment_values == 0
    treated_mask = treatment_values == 1
    X_control = X.iloc[control_mask] if isinstance(X, pd.DataFrame) else X[control_mask]
    X_treated = X.iloc[treated_mask] if isinstance(X, pd.DataFrame) else X[treated_mask]
    model_control.fit(X_control, outcome_values[control_mask])
    model_treated.fit(X_treated, outcome_values[treated_mask])

    def churn_probability(model):
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError("Each treatment arm must contain churn outcomes")
        return model.predict_proba(X)[:, classes.index(1)]

    p_control = churn_probability(model_control)
    p_treated = churn_probability(model_treated)
    return p_control - p_treated, model_control, model_treated


def fit_t_learner(frame: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Score an honest holdout; positive uplift means predicted prevented churn."""

    _validate_experiment(frame)
    id_columns = {"campaign_row_id", "y", "t"}
    features = [column for column in frame.columns if column not in id_columns]
    if not features:
        raise ValueError("The campaign table has no covariates for uplift modeling")
    X = frame[features].replace([np.inf, -np.inf], np.nan)
    strata = frame[["t", "y"]].astype(str).agg("_".join, axis=1)
    train_idx, test_idx = train_test_split(
        frame.index,
        test_size=0.30,
        random_state=random_state,
        stratify=strata,
    )
    treated_train = frame.loc[train_idx, "t"].eq(1)
    model_t = _campaign_pipeline(X.loc[train_idx], random_state)
    model_c = _campaign_pipeline(X.loc[train_idx], random_state + 1)
    model_t.fit(
        X.loc[train_idx][treated_train],
        frame.loc[train_idx, "y"][treated_train].astype(int),
    )
    model_c.fit(
        X.loc[train_idx][~treated_train],
        frame.loc[train_idx, "y"][~treated_train].astype(int),
    )

    report_columns = [
        column for column in ["campaign_row_id", "t", "y"] if column in frame
    ]
    scored = frame.loc[test_idx, report_columns].copy()
    scored["p_churn_treated"] = model_t.predict_proba(X.loc[test_idx])[:, 1]
    scored["p_churn_control"] = model_c.predict_proba(X.loc[test_idx])[:, 1]
    scored["predicted_uplift"] = (
        scored["p_churn_control"] - scored["p_churn_treated"]
    )
    return scored.sort_values("predicted_uplift", ascending=False).reset_index(drop=True)


def econml_causal_forest(
    X: np.ndarray | pd.DataFrame,
    treatment: np.ndarray | pd.Series,
    outcome: np.ndarray | pd.Series,
    random_state: int = 42,
):
    """Optional causal-forest estimate; positive output means prevented churn."""

    try:
        from econml.dml import CausalForestDML
    except ImportError as exc:
        raise ImportError("Install econml to use econml_causal_forest") from exc

    estimator = CausalForestDML(
        discrete_treatment=True,
        n_estimators=500,
        min_samples_leaf=10,
        random_state=random_state,
    )
    estimator.fit(Y=np.asarray(outcome), T=np.asarray(treatment), X=np.asarray(X))
    return -estimator.effect(np.asarray(X)), estimator


def uplift_by_decile(scored: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    ranked = scored.sort_values("predicted_uplift", ascending=False).copy()
    effective_bins = min(bins, len(ranked))
    ranked["uplift_decile"] = pd.qcut(
        np.arange(len(ranked)), effective_bins, labels=range(1, effective_bins + 1)
    )
    rows = []
    for decile, group in ranked.groupby("uplift_decile", observed=True):
        treated = group.loc[group["t"].eq(1), "y"]
        control = group.loc[group["t"].eq(0), "y"]
        rows.append(
            {
                "uplift_decile": int(decile),
                "customers": len(group),
                "mean_predicted_uplift": group["predicted_uplift"].mean(),
                "control_churn_rate": control.mean(),
                "treated_churn_rate": treated.mean(),
                "observed_churn_reduction": control.mean() - treated.mean(),
            }
        )
    return pd.DataFrame(rows)


def qini_like_curve(scored: pd.DataFrame) -> pd.DataFrame:
    """Create an inverse-propensity adjusted Qini diagnostic on holdout data."""

    ranked = scored.sort_values("predicted_uplift", ascending=False).reset_index(
        drop=True
    )
    propensity = ranked["t"].mean()
    if propensity <= 0 or propensity >= 1:
        raise ValueError("Qini evaluation requires treated and control rows")
    incremental = (
        ranked["y"] * (1 - ranked["t"]) / (1 - propensity)
        - ranked["y"] * ranked["t"] / propensity
    )
    ranked["target_fraction"] = (np.arange(len(ranked)) + 1) / len(ranked)
    ranked["cumulative_incremental_gain"] = incremental.cumsum()
    ranked["random_gain"] = (
        ranked["cumulative_incremental_gain"].iloc[-1]
        * ranked["target_fraction"]
    )
    ranked["qini_gain"] = (
        ranked["cumulative_incremental_gain"] - ranked["random_gain"]
    )
    return ranked[
        [
            "target_fraction",
            "cumulative_incremental_gain",
            "random_gain",
            "qini_gain",
        ]
    ]


def qini_auc(curve: pd.DataFrame) -> float:
    """Area between the ranked targeting curve and its random-policy baseline."""

    return float(trapezoid(curve["qini_gain"], curve["target_fraction"]))


def expected_customer_value(
    churn_reduction,
    monthly_margin: float,
    expected_remaining_months: float,
    campaign_cost: float,
):
    """Expected incremental profit from targeting one customer."""

    return (
        np.asarray(churn_reduction) * monthly_margin * expected_remaining_months
        - campaign_cost
    )


def target_profitable_customers(
    scored: pd.DataFrame,
    *,
    monthly_margin: float,
    expected_remaining_months: float,
    campaign_cost: float,
) -> pd.DataFrame:
    """Attach expected campaign value and an economically justified target flag."""

    result = scored.copy()
    result["expected_net_value"] = expected_customer_value(
        result["predicted_uplift"],
        monthly_margin,
        expected_remaining_months,
        campaign_cost,
    )
    result["target"] = result["expected_net_value"] > 0
    return result.sort_values("expected_net_value", ascending=False)


def optimize_threshold(
    uplift,
    retained_customer_value: float,
    campaign_cost: float,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compare uplift thresholds by total expected incremental profit."""

    uplift_values = np.asarray(uplift, dtype=float)
    if thresholds is None:
        thresholds = np.unique(np.quantile(uplift_values, np.linspace(0, 1, 101)))
    rows = []
    for threshold in thresholds:
        selected = uplift_values >= threshold
        net = uplift_values[selected] * retained_customer_value - campaign_cost
        rows.append(
            {
                "uplift_threshold": float(threshold),
                "customers_targeted": int(selected.sum()),
                "expected_total_profit": float(net.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "expected_total_profit", ascending=False
    ).reset_index(drop=True)


def posterior_campaign_profit(
    reduction_samples: np.ndarray,
    n_targeted: int,
    monthly_margin: float,
    retained_months: float,
    cost_per_customer: float,
) -> pd.DataFrame:
    """Translate posterior churn reduction into campaign-profit uncertainty."""

    samples = (
        np.asarray(reduction_samples)
        * n_targeted
        * monthly_margin
        * retained_months
        - n_targeted * cost_per_customer
    )
    low, high = np.quantile(samples, [0.025, 0.975])
    return pd.DataFrame(
        [
            {
                "expected_profit": samples.mean(),
                "profit_ci_95_low": low,
                "profit_ci_95_high": high,
                "probability_profitable": (samples > 0).mean(),
            }
        ]
    )


def _policy_gain(group: pd.DataFrame) -> float:
    propensity = group["t"].mean()
    if propensity <= 0 or propensity >= 1:
        return np.nan
    prevented = (
        group["y"] * (1 - group["t"]) / (1 - propensity)
        - group["y"] * group["t"] / propensity
    )
    return float(prevented.sum())


def compare_policies(
    scored: pd.DataFrame,
    random_state: int = 42,
    *,
    retained_customer_value: float = 480.0,
    campaign_cost: float = 10.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    work = scored.copy()
    work["random_score"] = rng.random(len(work))
    rows = []
    for fraction in [0.10, 0.20, 0.30, 0.50, 1.00]:
        n = max(1, int(len(work) * fraction))
        for policy, score in {
            "uplift": "predicted_uplift",
            "churn_risk": "p_churn_control",
            "random": "random_score",
        }.items():
            selected = work.nlargest(n, score)
            gain = _policy_gain(selected)
            rows.append(
                {
                    "policy": policy,
                    "target_fraction": fraction,
                    "customers_targeted": n,
                    "estimated_churns_prevented": gain,
                    "estimated_incremental_profit": (
                        gain * retained_customer_value - n * campaign_cost
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_experiment_analysis(
    frame: pd.DataFrame,
    random_state: int = 42,
    *,
    retained_customer_value: float = 480.0,
    campaign_cost: float = 10.0,
) -> ExperimentAnalysis:
    """Run population A/B inference, holdout uplift validation, and policy value."""

    summary = estimate_ab_effect(frame)
    bayesian = bayesian_ab_churn(frame, seed=random_state)
    scored = fit_t_learner(frame, random_state)
    deciles = uplift_by_decile(scored)
    qini = qini_like_curve(scored)
    policies = compare_policies(
        scored,
        random_state,
        retained_customer_value=retained_customer_value,
        campaign_cost=campaign_cost,
    )
    return ExperimentAnalysis(
        summary, bayesian.summary, scored, deciles, qini, policies
    )
