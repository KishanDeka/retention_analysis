import numpy as np
from src.decision import bayesian_ab_churn, expected_customer_value, optimize_threshold

def test_bayesian_ab_direction():
    r = bayesian_ab_churn(320,1000,260,1000,draws=10000,seed=1)
    assert r["prob_treatment_better"] > 0.9
    assert r["expected_absolute_reduction"] > 0

def test_expected_value():
    ev = expected_customer_value(
        np.array([0.1]), np.array([40.0]), np.array([12.0]), np.array([30.0])
    )
    assert np.isclose(ev[0], 18.0)

def test_optimize_threshold():
    best, _ = optimize_threshold(
        uplift=np.array([0.01,0.05,0.20]),
        value=np.array([500,500,500]),
        cost=np.array([30,30,30])
    )
    assert best[2] >= 1
