-- Treatment vs control campaign performance
SELECT
    CASE WHEN treatment = 1 THEN 'Treatment' ELSE 'Control' END AS experiment_group,
    COUNT(*) AS customers,
    SUM(churn_after_campaign) AS churned_customers,
    ROUND(100.0 * AVG(churn_after_campaign), 2) AS churn_rate_pct,
    ROUND(SUM(campaign_cost), 2) AS campaign_cost
FROM retention_experiment
GROUP BY treatment;

-- Absolute churn difference
WITH rates AS (
    SELECT
        treatment,
        AVG(churn_after_campaign) AS churn_rate
    FROM retention_experiment
    GROUP BY treatment
)
SELECT
    c.churn_rate AS control_churn_rate,
    t.churn_rate AS treatment_churn_rate,
    c.churn_rate - t.churn_rate AS absolute_churn_reduction
FROM rates c
JOIN rates t
  ON c.treatment = 0 AND t.treatment = 1;
