import numpy as np
from scipy.stats import beta

def beta_posterior(successes, failures, alpha_prior=1.0, beta_prior=1.0):
    return alpha_prior + successes, beta_prior + failures

def bayesian_ab_churn(
    churn_control, n_control, churn_treat, n_treat,
    draws=100_000, seed=42
):
    rng = np.random.default_rng(seed)
    ac, bc = beta_posterior(churn_control, n_control - churn_control)
    at, bt = beta_posterior(churn_treat, n_treat - churn_treat)
    pc = rng.beta(ac, bc, draws)
    pt = rng.beta(at, bt, draws)
    reduction = pc - pt
    return {
        "control_samples": pc,
        "treatment_samples": pt,
        "reduction_samples": reduction,
        "prob_treatment_better": float(np.mean(reduction > 0)),
        "expected_absolute_reduction": float(np.mean(reduction)),
        "credible_interval_95": tuple(np.quantile(reduction, [0.025, 0.975]))
    }

def expected_customer_value(
    churn_reduction,
    monthly_margin,
    expected_remaining_months,
    campaign_cost
):
    retained_value = np.asarray(churn_reduction) * np.asarray(monthly_margin) * np.asarray(expected_remaining_months)
    return retained_value - np.asarray(campaign_cost)

def target_profitable_customers(
    customer_ids,
    churn_reduction,
    monthly_margin,
    expected_remaining_months,
    campaign_cost
):
    ev = expected_customer_value(
        churn_reduction, monthly_margin, expected_remaining_months, campaign_cost
    )
    return [
        {"customer_id": cid, "expected_net_value": float(v), "target": bool(v > 0)}
        for cid, v in zip(customer_ids, ev)
    ]

def optimize_threshold(uplift, value, cost, thresholds=None):
    uplift = np.asarray(uplift)
    value = np.asarray(value)
    cost = np.asarray(cost)
    if thresholds is None:
        thresholds = np.quantile(uplift, np.linspace(0, 1, 101))
    results = []
    for t in thresholds:
        mask = uplift >= t
        profit = np.sum(uplift[mask] * value[mask] - cost[mask])
        results.append((float(t), float(profit), int(mask.sum())))
    return max(results, key=lambda x: x[1]), results

def posterior_campaign_profit(
    reduction_samples,
    n_targeted,
    monthly_margin,
    retained_months,
    cost_per_customer
):
    saved_customers = np.asarray(reduction_samples) * n_targeted
    profit = saved_customers * monthly_margin * retained_months - n_targeted * cost_per_customer
    return {
        "expected_profit": float(np.mean(profit)),
        "prob_profitable": float(np.mean(profit > 0)),
        "credible_interval_95": tuple(np.quantile(profit, [0.025, 0.975])),
        "profit_samples": profit
    }
