-- Clean and validate the full Orange Belgium randomized retention campaign.
-- run:
-- psql -U "$USER" -d customer_retention -f sql/01_clean_orange_campaign.sql

BEGIN;

DROP TABLE IF EXISTS orange_campaign;

CREATE TABLE orange_campaign ();

-- Add PC1-PC160 in the same order as the CSV.
DO $$
DECLARE
    i INTEGER;
BEGIN
    FOR i IN 1..160 LOOP
        EXECUTE format(
            'ALTER TABLE orange_campaign ADD COLUMN pc%s TEXT',
            i
        );
    END LOOP;
END $$;

-- Add FACTOR1-FACTOR18.
DO $$
DECLARE
    i INTEGER;
BEGIN
    FOR i IN 1..18 LOOP
        EXECUTE format(
            'ALTER TABLE orange_campaign ADD COLUMN factor%s TEXT',
            i
        );
    END LOOP;
END $$;

ALTER TABLE orange_campaign
    ADD COLUMN y TEXT,
    ADD COLUMN t TEXT;

-- Must remain one line because \copy is a psql command.
\copy orange_campaign FROM 'data/raw/churn_uplift_orange.csv' WITH (FORMAT CSV, HEADER TRUE, ENCODING 'UTF8');

ALTER TABLE orange_campaign
    ADD COLUMN campaign_row_id BIGSERIAL;

DO $$
DECLARE
    col_name TEXT;
BEGIN
    FOR col_name IN
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'orange_campaign'
          AND data_type = 'text'
    LOOP
        EXECUTE format(
            'UPDATE orange_campaign
             SET %I = NULLIF(BTRIM(%I), '''')',
            col_name,
            col_name
        );
    END LOOP;
END $$;

-- Remove invalid treatment/outcome records.
DELETE FROM orange_campaign
WHERE y IS NULL
   OR t IS NULL
   OR y NOT IN ('0', '1')
   OR t NOT IN ('0', '1');


-- Convert PC1-PC160 to numerical columns.
-- Existing NULL values remain NULL.
DO $$
DECLARE
    alter_clauses TEXT;
BEGIN
    SELECT string_agg(
        format(
            'ALTER COLUMN pc%s
             TYPE DOUBLE PRECISION
             USING pc%s::DOUBLE PRECISION',
            i,
            i
        ),
        ', '
        ORDER BY i
    )
    INTO alter_clauses
    FROM generate_series(1, 160) AS i;

    EXECUTE
        'ALTER TABLE orange_campaign '
        || alter_clauses;
END $$;


-- Convert and constrain the experimental variables.
ALTER TABLE orange_campaign
    ALTER COLUMN y TYPE SMALLINT
        USING y::SMALLINT,
    ALTER COLUMN t TYPE SMALLINT
        USING t::SMALLINT,
    ALTER COLUMN y SET NOT NULL,
    ALTER COLUMN t SET NOT NULL;


ALTER TABLE orange_campaign
    ADD CONSTRAINT orange_campaign_pk
        PRIMARY KEY (campaign_row_id),
    ADD CONSTRAINT orange_valid_outcome
        CHECK (y IN (0, 1)),
    ADD CONSTRAINT orange_valid_treatment
        CHECK (t IN (0, 1));


COMMIT;

-- Minimal post-load validation
SELECT
    COUNT(*) AS imported_rows,
    COUNT(DISTINCT campaign_row_id) AS unique_rows,
    COUNT(*) FILTER (WHERE t = 0) AS control_customers,
    COUNT(*) FILTER (WHERE t = 1) AS treated_customers,
    COUNT(*) FILTER (WHERE y = 1) AS churned_customers,
    ROUND(100.0 * AVG(y), 2) AS churn_rate_pct
FROM orange_campaign;
