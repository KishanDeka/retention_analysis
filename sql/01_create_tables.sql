DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS subscriptions;
DROP TABLE IF EXISTS services;
DROP TABLE IF EXISTS churn;
DROP TABLE IF EXISTS retention_experiment;

CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    gender TEXT,
    senior_citizen INTEGER,
    partner TEXT,
    dependents TEXT,
    tenure INTEGER
);

CREATE TABLE subscriptions (
    customer_id TEXT,
    contract_type TEXT,
    paperless_billing TEXT,
    payment_method TEXT,
    monthly_charges REAL,
    total_charges REAL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE services (
    customer_id TEXT,
    internet_service TEXT,
    phone_service TEXT,
    online_security TEXT,
    tech_support TEXT,
    streaming_tv TEXT,
    streaming_movies TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE churn (
    customer_id TEXT,
    churn_flag INTEGER,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE retention_experiment (
    customer_id TEXT,
    experiment_group TEXT,
    treatment INTEGER,
    churn_after_campaign INTEGER,
    campaign_cost REAL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
