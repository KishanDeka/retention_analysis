import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

def t_learner_uplift(X, treatment, outcome, base_model=None):
    """
    Simple T-learner for binary outcomes.
    Returns estimated reduction in churn from treatment:
    P(churn|control, X) - P(churn|treatment, X)
    """
    if base_model is None:
        base_model = RandomForestRegressor(
            n_estimators=120, min_samples_leaf=20, random_state=42
        )

    model_control = clone(base_model)
    model_treat = clone(base_model)

    mask_c = np.asarray(treatment) == 0
    mask_t = np.asarray(treatment) == 1

    model_control.fit(X[mask_c], np.asarray(outcome)[mask_c])
    model_treat.fit(X[mask_t], np.asarray(outcome)[mask_t])

    p_control = np.clip(model_control.predict(X), 0, 1)
    p_treat = np.clip(model_treat.predict(X), 0, 1)
    uplift = p_control - p_treat
    return uplift, model_control, model_treat

def econml_causal_forest(X, treatment, outcome):
    """
    Optional EconML implementation.
    """
    try:
        from econml.dml import CausalForestDML
    except ImportError as e:
        raise ImportError("Install econml to use econml_causal_forest") from e

    est = CausalForestDML(
        discrete_treatment=True,
        n_estimators=500,
        min_samples_leaf=10,
        random_state=42
    )
    est.fit(Y=np.asarray(outcome), T=np.asarray(treatment), X=X)
    effect = est.effect(X)
    # If outcome=churn, negative effect means treatment lowers churn.
    churn_reduction = -effect
    return churn_reduction, est

def qini_like_curve(uplift, treatment, outcome):
    """
    Lightweight diagnostic: sort by predicted uplift and calculate
    cumulative observed difference in churn between control and treatment.
    Not a full production Qini implementation.
    """
    order = np.argsort(-np.asarray(uplift))
    t = np.asarray(treatment)[order]
    y = np.asarray(outcome)[order]
    rows = []
    for k in np.linspace(100, len(order), min(20, len(order)//100), dtype=int):
        tk, yk = t[:k], y[:k]
        tc = yk[tk==0].mean() if np.any(tk==0) else np.nan
        tt = yk[tk==1].mean() if np.any(tk==1) else np.nan
        rows.append((k, tc - tt))
    return rows
