-- run from root directory :
-- psql -U $USER -d telco_customer_db -f src/00_load_raw_data.sql
-- --------------------------------------------------------------
-- Purpose:
--   1. Create a raw staging table for the merged IBM Telco
--      Customer Churn (11.1.3+) dataset.
--   2. Load data/raw/customer_retention.csv with psql \copy.
--   3. Create a typed/cleaned view for downstream SQL analysis.
-- -------------------------------------------------------------
-- Drop old staging objects so the load is reproducible
-- ------------------------------------------------------------

DROP TABLE IF EXISTS customer_retention;

-- Text ingestion keeps CSV loading robust.

CREATE TABLE customer_retention (
    customer_id                       TEXT,
    gender                            TEXT,
    age                               TEXT,
    under_30                          TEXT,
    senior_citizen                    TEXT,
    married                           TEXT,
    dependents                        TEXT,
    number_of_dependents              TEXT,
    country                           TEXT,
    state                             TEXT,
    city                              TEXT,
    zip_code                          TEXT,
    latitude                          TEXT,
    longitude                         TEXT,
    population                        TEXT,
    quarter                           TEXT,
    referred_a_friend                 TEXT,
    number_of_referrals               TEXT,
    tenure_in_months                  TEXT,
    offer                             TEXT,
    phone_service                     TEXT,
    avg_monthly_long_distance_charges TEXT,
    multiple_lines                    TEXT,
    internet_service                  TEXT,
    internet_type                     TEXT,
    avg_monthly_gb_download           TEXT,
    online_security                   TEXT,
    online_backup                     TEXT,
    device_protection_plan            TEXT,
    premium_tech_support              TEXT,
    streaming_tv                      TEXT,
    streaming_movies                  TEXT,
    streaming_music                   TEXT,
    unlimited_data                    TEXT,
    contract                          TEXT,
    paperless_billing                 TEXT,
    payment_method                    TEXT,
    monthly_charge                    TEXT,
    total_charges                     TEXT,
    total_refunds                     TEXT,
    total_extra_data_charges          TEXT,
    total_long_distance_charges       TEXT,
    total_revenue                     TEXT,
    satisfaction_score                TEXT,
    customer_status                   TEXT,
    churn_label                       TEXT,
    churn_score                       TEXT,
    cltv                              TEXT,
    churn_category                    TEXT,
    churn_reason                      TEXT
);

\copy customer_retention FROM 'data/raw/telco_customer_ibm.csv' WITH (FORMAT CSV, HEADER TRUE, ENCODING 'UTF8');

-- Trim all text fields and standardize blanks to NULL.
BEGIN;

-- ------------------------------------------------------------
-- Clean all TEXT columns
-- Trim whitespace and convert empty strings to NULL
-- ------------------------------------------------------------

DO $$
DECLARE
    col_name TEXT;
BEGIN
    FOR col_name IN
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'customer_retention'
          AND data_type = 'text'
    LOOP
        EXECUTE format(
            'UPDATE customer_retention
             SET %I = NULLIF(BTRIM(%I), '''')',
            col_name,
            col_name
        );
    END LOOP;
END $$;


-- ------------------------------------------------------------
-- Remove records that cannot have a valid primary key
-- ------------------------------------------------------------

DELETE FROM customer_retention
WHERE customer_id IS NULL;


-- Keep one row per customer_id
DELETE FROM customer_retention AS a
USING customer_retention AS b
WHERE a.customer_id = b.customer_id
  AND a.ctid < b.ctid;


-- ------------------------------------------------------------
-- Convert numerical columns
-- ------------------------------------------------------------

ALTER TABLE customer_retention
    ALTER COLUMN age
        TYPE INTEGER
        USING age::INTEGER,

    ALTER COLUMN number_of_dependents
        TYPE INTEGER
        USING number_of_dependents::INTEGER,

    ALTER COLUMN latitude
        TYPE DOUBLE PRECISION
        USING latitude::DOUBLE PRECISION,

    ALTER COLUMN longitude
        TYPE DOUBLE PRECISION
        USING longitude::DOUBLE PRECISION,

    ALTER COLUMN population
        TYPE INTEGER
        USING population::INTEGER,

    ALTER COLUMN number_of_referrals
        TYPE INTEGER
        USING number_of_referrals::INTEGER,

    ALTER COLUMN tenure_in_months
        TYPE INTEGER
        USING tenure_in_months::INTEGER,

    ALTER COLUMN avg_monthly_long_distance_charges
        TYPE NUMERIC(12,2)
        USING avg_monthly_long_distance_charges::NUMERIC(12,2),

    ALTER COLUMN avg_monthly_gb_download
        TYPE INTEGER
        USING avg_monthly_gb_download::INTEGER,

    ALTER COLUMN monthly_charge
        TYPE NUMERIC(12,2)
        USING monthly_charge::NUMERIC(12,2),

    ALTER COLUMN total_charges
        TYPE NUMERIC(14,2)
        USING total_charges::NUMERIC(14,2),

    ALTER COLUMN total_refunds
        TYPE NUMERIC(12,2)
        USING total_refunds::NUMERIC(12,2),

    ALTER COLUMN total_extra_data_charges
        TYPE NUMERIC(12,2)
        USING total_extra_data_charges::NUMERIC(12,2),

    ALTER COLUMN total_long_distance_charges
        TYPE NUMERIC(14,2)
        USING total_long_distance_charges::NUMERIC(14,2),

    ALTER COLUMN total_revenue
        TYPE NUMERIC(14,2)
        USING total_revenue::NUMERIC(14,2),

    ALTER COLUMN satisfaction_score
        TYPE INTEGER
        USING satisfaction_score::INTEGER,

    ALTER COLUMN churn_score
        TYPE INTEGER
        USING churn_score::INTEGER,

    ALTER COLUMN cltv
        TYPE NUMERIC(14,2)
        USING cltv::NUMERIC(14,2);


ALTER TABLE customer_retention
ADD COLUMN IF NOT EXISTS churn_value SMALLINT GENERATED ALWAYS AS (
    CASE 
        WHEN LOWER(churn_label) = 'yes' THEN 1
        WHEN LOWER(churn_label) = 'no'  THEN 0
        ELSE NULL
    END
) STORED;

-- ------------------------------------------------------------
-- Add data-integrity constraints
-- ------------------------------------------------------------

ALTER TABLE customer_retention
    ALTER COLUMN customer_id SET NOT NULL;

ALTER TABLE customer_retention
    ADD CONSTRAINT customer_retention_pk
        PRIMARY KEY (customer_id);

ALTER TABLE customer_retention
    ADD CONSTRAINT valid_churn_value
        CHECK (churn_value IN (0, 1));


COMMIT;

-- Compact data-quality check.
SELECT
    COUNT(*) AS customers,
    COUNT(*) FILTER (WHERE total_charges IS NULL) AS missing_total_charges,
    COUNT(*) FILTER (WHERE churn_value = 1) AS churned_customers,
    ROUND(100.0 * AVG(churn_value), 2) AS churn_rate_pct
FROM customer_retention;
