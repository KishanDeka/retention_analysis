WITH customer_features AS (
    SELECT
        c.customer_id,
        c.tenure,
        s.contract_type,
        s.payment_method,
        s.monthly_charges,
        sv.tech_support,
        ch.churn_flag,
        CASE
            WHEN s.contract_type = 'Month-to-month' THEN 3 ELSE 0
        END
        + CASE WHEN c.tenure < 12 THEN 2 ELSE 0 END
        + CASE WHEN sv.tech_support = 'No' THEN 1 ELSE 0 END
        + CASE WHEN s.monthly_charges >= 85 THEN 1 ELSE 0 END
        + CASE WHEN s.payment_method = 'Electronic check' THEN 1 ELSE 0 END
        AS risk_score
    FROM customers c
    JOIN subscriptions s USING (customer_id)
    JOIN services sv USING (customer_id)
    JOIN churn ch USING (customer_id)
)
SELECT *,
       CASE
           WHEN risk_score >= 6 THEN 'High'
           WHEN risk_score >= 3 THEN 'Medium'
           ELSE 'Low'
       END AS risk_segment
FROM customer_features;
