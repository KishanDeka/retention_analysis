-- Descriptive churn analysis only; no tables or views are created.

-- Overall churn and revenue at risk.
SELECT
    COUNT(*) AS customers,
    SUM(churn_value) AS churned_customers,
    ROUND(100.0 * AVG(churn_value), 2) AS churn_rate_pct,
    ROUND(SUM(monthly_charge) FILTER (WHERE churn_value = 1), 2) AS monthly_revenue_at_risk
FROM customer_retention;

-- Churn by contract, with a minimum-size guard for useful comparisons.
SELECT
    contract,
    COUNT(*) AS customers,
    SUM(churn_value) AS churned_customers,
    ROUND(100.0 * AVG(churn_value), 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charge), 2) AS avg_monthly_charge
FROM customer_retention
GROUP BY contract
HAVING COUNT(*) >= 30
ORDER BY churn_rate_pct DESC;

-- Churn by tenure band.
WITH banded AS (
    SELECT *, CASE
        WHEN tenure_in_months <= 6 THEN '0-6 months'
        WHEN tenure_in_months <= 12 THEN '7-12 months'
        WHEN tenure_in_months <= 24 THEN '13-24 months'
        WHEN tenure_in_months <= 48 THEN '25-48 months'
        ELSE '49+ months'
    END AS tenure_band
    FROM customer_retention
)
SELECT
    tenure_band,
    COUNT(*) AS customers,
    ROUND(100.0 * AVG(churn_value), 2) AS churn_rate_pct,
    ROUND(SUM(monthly_charge) FILTER (WHERE churn_value = 1), 2) AS monthly_revenue_at_risk
FROM banded
GROUP BY tenure_band
ORDER BY MIN(tenure_in_months);

-- Ranked churn reasons. Churn reason is descriptive only and must not be a model feature.
SELECT churn_reason, COUNT(*) AS churned_customers
FROM customer_retention
WHERE churn_value = 1 AND churn_reason IS NOT NULL
GROUP BY churn_reason
ORDER BY churned_customers DESC;

-- Action-oriented customer segments, calculated as a query (no segment table).
WITH threshold AS (
    SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY monthly_charge) AS median_charge
    FROM customer_retention
), segmented AS (
    SELECT
        customer_id,
        churn_value,
        monthly_charge,
        tenure_in_months,
        contract,
        premium_tech_support,
        CASE
            WHEN tenure_in_months <= 6 AND LOWER(contract) = 'month-to-month'
                THEN 'new_and_fragile'
            WHEN monthly_charge >= median_charge
                 AND LOWER(contract) = 'month-to-month'
                THEN 'high_value_at_risk'
            WHEN LOWER(contract) = 'month-to-month' AND COALESCE(LOWER(premium_tech_support), 'no') <> 'yes'
                THEN 'unsupported_flexible'
            WHEN monthly_charge >= median_charge
                THEN 'high_value_loyal'
            ELSE 'core_customers'
        END AS customer_segment
    FROM customer_retention CROSS JOIN threshold
)
SELECT
    customer_segment,
    COUNT(*) AS customers,
    SUM(churn_value) AS churned_customers,
    ROUND(100.0 * AVG(churn_value), 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charge), 2) AS avg_monthly_charge,
    ROUND(SUM(monthly_charge) FILTER (WHERE churn_value = 1), 2) AS monthly_revenue_at_risk
FROM segmented
GROUP BY customer_segment
ORDER BY monthly_revenue_at_risk DESC NULLS LAST;

-- Orange full-sample A/B summary. Positive reduction means treatment helped.
SELECT
    t AS treatment,
    COUNT(*) AS customers,
    SUM(y) AS churned_customers,
    ROUND(100.0 * AVG(y), 3) AS churn_rate_pct
FROM orange_campaign
GROUP BY t
ORDER BY t;

-- Treatment effect expressed as control churn minus treated churn.
SELECT
    AVG(y) FILTER (WHERE t = 0) AS control_churn_rate,
    AVG(y) FILTER (WHERE t = 1) AS treated_churn_rate,
    AVG(y) FILTER (WHERE t = 0)
      - AVG(y) FILTER (WHERE t = 1) AS absolute_churn_reduction
FROM orange_campaign;

-- Customer-level uplift scores are intentionally evaluated in Python on a held-out split.
