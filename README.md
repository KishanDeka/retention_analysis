# Customer Retention Decision Analytics

**PostgreSQL · Power BI · PCA · Machine Learning · SHAP · A/B Testing · Uplift Modeling**

An end-to-end telecom retention project that connects descriptive analytics,
churn prediction, model explanation, randomized experimentation, and
profit-aware campaign targeting.

The project answers five business questions:

1. Where are churn and revenue exposure concentrated?
2. Which customers are most likely to churn?
3. Does PCA improve validation performance enough to justify its use?
4. Does a retention intervention causally reduce churn?
5. Which customers should be targeted to maximize expected incremental value?

> The two datasets represent complementary stages of a retention workflow.
> Their customer rows are not joined.

## Decision framework

```mermaid
flowchart TD
    A["IBM Telco data"] --> B["SQL analytics and Power BI"]
    A --> C["Churn prediction and SHAP"]
    D["Orange randomized campaign"] --> E["A/B test and uplift"]
    B --> F["Retention decision"]
    C --> F
    E --> F
```

Churn probability answers **who may leave**. Uplift answers **whose behavior
may change because of treatment**. A high-risk customer is not automatically a
valuable campaign target.

## Data

Raw data are not committed to the repository. Download the files and place
them in `data/raw/`.

| Dataset | Local filename | Used for | Source |
|---|---|---|---|
| IBM Telco Customer Churn 11.1.3+ | `telco_customer_churn.csv` | SQL analytics, Power BI, customer segments, PCA, churn models, SHAP | [IBM dataset description](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113) · [Kaggle download mirror](https://www.kaggle.com/datasets/ylchang/telco-customer-churn-1113) |
| Orange Belgium Churn Uplift | `churn_uplift_anonymized.csv` | Randomized A/B testing, T-learner uplift, Qini evaluation, campaign value | [Authors' benchmark repository and download](https://github.com/TheoVerhelst/Churn-Uplift-Dataset-Paper) · [Associated paper](https://arxiv.org/abs/2312.07206) |

IBM describes its sample as a fictional California telecom company with 7,043
customers and interpretable demographic, service, financial, satisfaction,
churn, and customer-lifetime-value fields. The Orange Belgium dataset contains
anonymized covariates, a binary treatment indicator `t`, and a binary churn
outcome `y` from a real retention-campaign setting.

The datasets are used separately:

- **IBM Telco:** business interpretation and churn-risk prediction.
- **Orange Belgium:** causal campaign evaluation and treatment-aware targeting.

## Repository structure

```text
customer-retention-analytics/
├── README.md
├── requirements.txt
├── data/
│   └── raw/
│       ├── telco_customer_churn.csv
│       └── churn_uplift_anonymized.csv
├── sql/
│   ├── 00_load_raw_data.sql
│   └── 01_clean_orange_campaign.sql
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── churn.py
│   └── experiment.py
├── notebooks/
│   ├── 01_telco_eda_segments.ipynb
│   ├── 02_churn_models_pca.ipynb
│   ├── 03_shap_interpretation.ipynb
│   └── 04_ab_testing_uplift.ipynb
├── outputs/
│   ├── figures/
│   └── *.csv
├── powerbi/
│   └── customer_retention_dashboard.pbix
└── tests/
    ├── test_data.py
    ├── test_churn.py
    └── test_experiment.py
```

Reusable logic belongs in `src/`. The notebooks contain analysis,
visualization, and interpretation rather than duplicate implementations.
Earlier `features.py`, `causal.py`, and `decision.py` modules are no longer
needed: their retained functionality is consolidated into `churn.py`,
`models.py`, and `experiment.py`.

## Python modules

| Module | Responsibility |
|---|---|
| `src/data.py` | Load the SQL-cleaned Telco and Orange tables and validate their minimum schemas |
| `src/models.py` | Shared preprocessing and individual logistic, PCA-logistic, random-forest, gradient-boosting, XGBoost, and LightGBM definitions |
| `src/churn.py` | Business features, leakage protection, cross-validation, holdout scoring, PCA comparison, permutation importance, and SHAP helpers |
| `src/experiment.py` | Frequentist and Bayesian A/B tests, T-learner uplift, optional causal forest, Qini analysis, targeting policies, and profit calculations |

SQL remains the source of truth for ingestion, type conversion, blank-to-null
handling, constraints, and load validation. Python performs only analytical
feature preparation and modeling.

## Required SQL schema contract

The notebooks expect these PostgreSQL objects:

| Object | Required fields |
|---|---|
| `public.customer_retention` | `customer_id`, `churn_value`, `tenure_in_months`, `monthly_charge`; `cltv`, `contract`, and `premium_tech_support` are used when available |
| `public.orange_campaign` | `campaign_row_id`, `t`, `y`, plus campaign covariates such as `pc1`–`pc160` and `factor1`–`factor18` |

Keep these names consistent in SQL rather than adding a second cleaning or
renaming layer in Python. In particular, older IBM extracts sometimes use
`tenure_months`, `monthly_charges`, and `tech_support`; alias or rename them in
the SQL output to the contract above.

## Setup

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create the PostgreSQL database from the repository root:

```bash
createdb customer_retention

psql -U "$USER" -d customer_retention \
  -f sql/00_load_raw_data.sql

psql -U "$USER" -d customer_retention \
  -f sql/01_clean_orange_campaign.sql
```

Set the connection string:

```bash
export DATABASE_URL="postgresql+psycopg://USER:PASSWORD@localhost/customer_retention"
```

Confirm the database contract:

```sql
SELECT COUNT(*) FROM customer_retention;
SELECT COUNT(*) FROM orange_campaign;

SELECT customer_id, churn_value, tenure_in_months, monthly_charge
FROM customer_retention
LIMIT 5;

SELECT campaign_row_id, t, y
FROM orange_campaign
LIMIT 5;
```

## Run the analysis

Start Jupyter from the repository root:

```bash
jupyter lab
```

Run the notebooks in order:

| Notebook | Main output |
|---|---|
| `01_telco_eda_segments.ipynb` | Churn KPIs, customer segments, and revenue exposure |
| `02_churn_models_pca.ipynb` | Cross-validated model leaderboard, PCA validation, calibration, and customer churn scores |
| `03_shap_interpretation.ipynb` | Global and customer-level model explanations |
| `04_ab_testing_uplift.ipynb` | A/B effect, Bayesian uncertainty, uplift deciles, Qini curve, and targeting profit |

Each notebook loads data through `src.data`. The notebooks export the loaded SQL
tables and derived results to `outputs/` for reproducibility and Power BI.

## Churn modeling methodology

The churn workflow uses a stratified train/holdout split. Models are compared
with five-fold stratified cross-validation only inside the training partition;
the selected model is evaluated once on the untouched holdout set.

Default comparison:

- Logistic regression
- Logistic regression with PCA on numeric predictors
- Random forest
- Histogram gradient boosting

XGBoost and LightGBM builders are also available in `src/models.py` and can be
added to the default registry.

Model selection emphasizes:

- **PR-AUC:** primary ranking metric for the minority churn class.
- **ROC-AUC:** supporting overall ranking metric.
- **Brier score and calibration:** quality of churn probabilities.
- **Precision and recall:** operational performance at a chosen threshold.

PCA is retained only if it provides measurable cross-validated benefit,
improved stability, or useful compression. Similar performance favors the
non-PCA model because original features are easier to interpret.

Post-outcome and externally generated fields are excluded from model inputs,
including `churn_label`, `customer_status`, `churn_reason`, `churn_category`,
and IBM's precomputed `churn_score`.

## Experiment and uplift methodology

The Orange workstream first estimates the average randomized treatment effect:

$$
ATE=P(Y=1\mid T=0)-P(Y=1\mid T=1)
$$

A positive value means treatment reduced churn. The A/B report contains group
sizes, churn rates, absolute and relative reduction, a 95% confidence interval,
a two-proportion z-test, number needed to treat, and a Bayesian probability
that treatment is beneficial.

The T-learner then estimates customer-level churn reduction:

$$
\hat{\tau}(x)=\hat{P}(Y=1\mid T=0,X=x)-\hat{P}(Y=1\mid T=1,X=x)
$$

Uplift performance is evaluated on held-out customers using uplift deciles, a
Qini-style curve, and policy comparisons against churn-risk and random
targeting. A complex targeting policy should only be adopted when its
out-of-sample incremental gain is stable.

The final decision rule is economic:

$$
Expected\ Net\ Value_i=\hat{\tau}(x_i)\times Retained\ Customer\ Value_i-Campaign\ Cost_i
$$

## Business outputs and insights

The project turns model outputs into decisions rather than stopping at model
accuracy.

| Analysis | Business interpretation | Decision supported |
|---|---|---|
| Churn and revenue by segment | A segment can have high churn but limited financial exposure, or moderate churn with substantial value at risk | Prioritize segments using both churn and value |
| Calibrated churn probability | Customers can be ranked by expected risk instead of a hard yes/no label | Allocate finite retention capacity |
| PCA validation | Dimensionality reduction is useful only if it improves validation or stability | Retain interpretability when PCA adds no measurable benefit |
| SHAP explanations | Predictive signals explain why a customer receives a high score | Design testable onboarding, support, contract, or pricing hypotheses |
| Randomized A/B effect | Treatment-control differences estimate whether the campaign works on average | Decide whether the intervention merits continued investment |
| Uplift and Qini | High-risk customers are not always the customers whose behavior treatment changes | Target persuadable customers rather than risk alone |
| Expected incremental value | Statistical uplift is translated using customer value and campaign cost | Select a profitable targeting threshold |

SHAP describes model associations; it does not establish that changing a
feature will prevent churn. Causal claims are restricted to the randomized
Orange campaign analysis.

## Figures for the GitHub README

Keep the final README focused. Four figures are enough and should be saved from
the executed notebooks to `outputs/figures/`.

| Priority | Figure | Suggested path | Why it belongs in the README |
|---:|---|---|---|
| 1 | Segment churn rate and revenue at risk | `outputs/figures/01_segment_value_risk.png` | Strongest link between descriptive analysis and a retention decision |
| 2 | Cross-validated PR-AUC/ROC-AUC plus PCA benefit | `outputs/figures/02_model_pca_comparison.png` | Demonstrates rigorous model selection and whether PCA added value |
| 3 | SHAP global summary or beeswarm | `outputs/figures/03_shap_summary.png` | Shows that the selected churn model is interpretable |
| 4 | Qini curve or campaign profit by target fraction | `outputs/figures/04_uplift_policy_value.png` | Demonstrates the transition from prediction to causal, profit-aware targeting |

Optional fifth figure: the control-versus-treatment churn-rate plot with a 95%
confidence interval. Include it only if the A/B result remains readable at
GitHub width.

After generating the files, place the four figures immediately after the
relevant Results paragraphs using relative links, for example:

```markdown
![Segment churn and revenue at risk](outputs/figures/01_segment_value_risk.png)
```

Do not publish figures with invented example values. Run the notebooks, save
the actual plots, and report the resulting metrics beside them.

## Power BI

Recommended report pages:

1. **Executive overview:** customers, churn rate, revenue at risk, and filters.
2. **Segments and drivers:** churn by contract, tenure, service, and business segment.
3. **Model and campaign decisions:** model performance, scored customers, A/B effect, uplift deciles, and policy value.

Primary dashboard inputs are generated under `outputs/`:

```text
customer_retention_sql.csv
customer_segment_summary.csv
model_leaderboard.csv
pca_validation.csv
customer_churn_scores.csv
shap_global_importance.csv
orange_campaign_sql.csv
ab_test_summary.csv
bayesian_ab_summary.csv
uplift_customer_scores.csv
uplift_deciles.csv
campaign_policy_results.csv
uplift_threshold_optimization.csv
```

Label IBM and Orange visuals clearly because they represent separate customer
populations.

## Tests

Run all tests from the repository root:

```bash
pytest -q
```

The tests should cover database-schema validation, churn leakage protection,
model construction, PCA comparison, A/B calculations, holdout uplift scoring,
Qini output, and policy reproducibility.

## Limitations

- IBM Telco has no randomized treatment assignment and cannot support causal
  retention claims.
- Orange covariates are anonymized, limiting feature-level business
  interpretation.
- Orange campaign conclusions apply to the experiment population and should not
  automatically be generalized to all IBM Telco customers.
- Individual uplift estimates are noisy and require honest holdout validation.
- Profit estimates depend on explicit customer-value and campaign-cost assumptions.
- A targeting policy selected offline should be validated in a new prospective
  experiment before production use.

## Reproducibility note

This README intentionally does not hard-code model scores or campaign lift.
Execute the SQL scripts and notebooks, then report the generated metrics and
figures. This prevents portfolio claims from drifting away from reproducible
outputs.

## Resume summary

> Built an end-to-end telecom retention decision workflow combining PostgreSQL
> analytics, Power BI segmentation, PCA validation, comparative churn modeling,
> SHAP explanations, randomized A/B testing, and uplift-based campaign targeting.
> Translated churn risk and treatment effects into profit-aware retention
> priorities using held-out validation.

## Data attribution

- IBM Cognos Analytics Samples Team, *Telco customer churn (11.1.3+)*.
- Verhelst, T., Mercier, D., Shrestha, J., and Bontempi, G., *A churn
  prediction dataset from the telecom sector: a new benchmark for uplift
  modeling*, arXiv:2312.07206.

Dataset terms remain governed by their respective sources. Do not redistribute
raw data unless the source license permits it.
