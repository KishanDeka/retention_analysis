-- Example cleaning checks after loading raw data.

SELECT COUNT(*) AS total_customers,
       COUNT(DISTINCT customer_id) AS unique_customers
FROM customers;

SELECT
    SUM(CASE WHEN tenure IS NULL THEN 1 ELSE 0 END) AS missing_tenure,
    SUM(CASE WHEN monthly_charges IS NULL THEN 1 ELSE 0 END) AS missing_monthly_charges,
    SUM(CASE WHEN total_charges IS NULL THEN 1 ELSE 0 END) AS missing_total_charges
FROM subscriptions s
JOIN customers c USING (customer_id);

-- Detect impossible / suspicious values.
SELECT *
FROM subscriptions
WHERE monthly_charges < 0 OR total_charges < 0;
