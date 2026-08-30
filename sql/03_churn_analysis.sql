-- Executive churn KPIs
SELECT
    COUNT(*) AS customers,
    SUM(churn_flag) AS churned_customers,
    ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct
FROM churn;

-- Churn by contract type
SELECT
    s.contract_type,
    COUNT(*) AS customers,
    SUM(c.churn_flag) AS churned,
    ROUND(100.0 * AVG(c.churn_flag), 2) AS churn_rate_pct,
    ROUND(AVG(s.monthly_charges), 2) AS avg_monthly_charges
FROM subscriptions s
JOIN churn c USING (customer_id)
GROUP BY s.contract_type
ORDER BY churn_rate_pct DESC;

-- Churn by tenure band
WITH base AS (
    SELECT
        c.customer_id,
        c.tenure,
        ch.churn_flag,
        CASE
            WHEN c.tenure <= 12 THEN '0-12'
            WHEN c.tenure <= 24 THEN '13-24'
            WHEN c.tenure <= 48 THEN '25-48'
            ELSE '49+'
        END AS tenure_group
    FROM customers c
    JOIN churn ch USING (customer_id)
)
SELECT
    tenure_group,
    COUNT(*) AS customers,
    ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct
FROM base
GROUP BY tenure_group
ORDER BY MIN(tenure);

-- Revenue at risk (simple monthly proxy)
SELECT
    ROUND(SUM(CASE WHEN ch.churn_flag = 1 THEN s.monthly_charges ELSE 0 END), 2)
        AS monthly_revenue_at_risk
FROM subscriptions s
JOIN churn ch USING (customer_id);
