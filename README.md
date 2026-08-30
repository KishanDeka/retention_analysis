# Telecom Customer Retention Analytics

An end-to-end analytics project for understanding customer churn, predicting churn risk, evaluating retention offers with Bayesian A/B testing, estimating customer-level treatment response, and optimizing retention targeting based on expected profit.

The project combines:

- SQL
- Python
- scikit-learn
- XGBoost / LightGBM
- Bayesian statistics
- SHAP explainability
- uplift modeling
- Power BI-ready reporting

---

## Business Problem

A telecom company wants to reduce customer churn without offering discounts to everyone.

The project answers four questions:

1. Which customers are most likely to churn?
2. What factors are driving churn?
3. Does a retention offer reduce churn?
4. Which customers should receive the offer to maximize expected profit?

---

## Project Workflow

```text
Raw Customer Data
        ↓
SQL Cleaning & Analysis
        ↓
EDA & Customer Segmentation
        ↓
PCA / Correlation Analysis
        ↓
Churn Prediction
        ↓
SHAP Interpretability
        ↓
Bayesian A/B Testing
        ↓
Uplift / Treatment Effect Modeling
        ↓
Expected Profit Optimization
        ↓
Power BI Reporting
```

---

## Repository Structure

```text
telco-retention-analytics/
│
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── data/
│   ├── generate_demo_data.py
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── sql/
│   ├── 01_create_tables.sql
│   ├── 02_data_cleaning.sql
│   ├── 03_churn_analysis.sql
│   ├── 04_customer_segments.sql
│   └── 05_experiment_analysis.sql
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_churn_drivers.ipynb
│   ├── 03_pca_analysis.ipynb
│   ├── 04_churn_model.ipynb
│   ├── 05_bayesian_ab_test.ipynb
│   └── 06_uplift_analysis.ipynb
│
├── src/
│   ├── features.py
│   ├── models.py
│   ├── causal.py
│   └── decision.py
│
├── powerbi/
│   └── README.md
│
├── reports/
│   └── figures/
│
└── tests/
```

---

## Data

The main churn dataset is the **IBM Telco Customer Churn** dataset.

Expected file:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The IBM dataset does not contain a randomized retention campaign.

For Bayesian A/B testing and uplift modeling, the project therefore includes a synthetic treatment/control experiment generated with:

```bash
python data/generate_demo_data.py
```

This allows the full analytical workflow to run without making unsupported causal claims from the IBM dataset.

---

## SQL Analysis

The SQL layer covers:

- customer and subscription tables
- data-quality checks
- overall churn rate
- churn by contract type
- churn by tenure
- churn by payment method
- revenue at risk
- customer risk segments
- treatment vs. control experiment metrics

This represents the core Data Analyst part of the project.

---

## Churn Prediction

The project estimates:

$$
P(\text{Churn}=1\mid X)
$$

using Logistic Regression as the interpretable baseline.

Optional models include:

- XGBoost
- LightGBM

Evaluation metrics include:

- ROC-AUC
- Precision-Recall AUC
- Brier Score

The output is a customer-level churn probability that can be used to create Low, Medium, and High risk segments.

---

## SHAP Interpretability

SHAP is used to explain the churn model and answer:

> Why is this customer predicted to churn?

For the tree-based models, SHAP can provide both global and individual explanations.

### Global interpretation

Use a SHAP summary plot to identify the most influential churn drivers.

Typical variables to examine include:

- contract type
- tenure
- monthly charges
- technical support
- payment method
- internet service

Example:

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)
```

This helps answer questions such as:

> Which factors contribute most strongly to churn predictions across the customer base?

### Individual customer interpretation

SHAP can also explain why a particular customer received a high churn score.

For example:

```text
Customer C001
Predicted churn probability: 78%

Main contributors:
+ Month-to-month contract
+ Low tenure
+ High monthly charges
+ No technical support
- Automatic payment
```

This makes the model easier to discuss with business stakeholders.

Recommended outputs:

```text
reports/figures/shap_summary.png
reports/figures/shap_bar.png
reports/figures/shap_customer_example.png
```

SHAP should be used for **model interpretation**, not interpreted as causal evidence.

---

## Bayesian A/B Testing

A simulated retention experiment compares:

```text
Control
    → no retention offer

Treatment
    → retention offer
```

For each group, churn probability is modeled with a Beta distribution.

The project estimates:

```text
Probability treatment reduces churn
Expected absolute churn reduction
95% credible interval
Expected campaign profit
Probability campaign is profitable
```

This provides a more decision-focused interpretation than reporting only a p-value.

---

## Uplift Modeling

A churn model answers:

> Who is likely to leave?

An uplift model answers:

> Whose churn probability is likely to decrease because of the retention offer?

The project estimates customer-level churn reduction as:

$$ U_i = P(\text{Churn} \mid T=0, X_i) - P(\text{Churn} \mid T=1, X_i) $$

A positive uplift means the offer is predicted to reduce that customer's churn probability.

The repository currently includes:

- T-Learner uplift estimation
- optional EconML Causal Forest implementation

---

## Financial Targeting

The final decision layer combines uplift with customer value and campaign cost.

For customer $i$:

$$ EV_i = U_i \times CLV_i - C_i $$

where:

- $U_i$ = predicted reduction in churn probability
- $CLV_i$ = estimated customer value
- $C_i$ = retention-offer cost

The basic decision rule is:

```text
Target customer if Expected Value > 0
```

This avoids targeting customers purely because they have high churn risk.

---

## Power BI

The project exports:

```text
data/processed/powerbi_customer_retention.csv
```

using:

```bash
python scripts_export_powerbi.py
```

Recommended Power BI pages:

### 1. Executive Overview

- total customers
- churn rate
- monthly revenue
- revenue at risk
- high-risk customers

### 2. Churn Drivers

- churn by contract
- churn by tenure
- churn by payment method
- churn by technical support
- customer risk segments

### 3. Retention Campaign

- control churn rate
- treatment churn rate
- expected churn reduction
- customers targeted
- campaign cost
- expected campaign profit

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/telco-retention-analytics.git
cd telco-retention-analytics
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate demo data:

```bash
python data/generate_demo_data.py
```

Run tests:

```bash
pytest
```

Start Jupyter:

```bash
jupyter notebook
```

---

## Notebook Order

Run:

```text
01_eda.ipynb
02_churn_drivers.ipynb
03_pca_analysis.ipynb
04_churn_model.ipynb
05_shap_interpretability.ipynb
06_bayesian_ab_test.ipynb
07_uplift_analysis.ipynb
```

---

## Main Skills Demonstrated

| Area | Skills |
|---|---|
| Analytics | SQL, EDA, KPIs, segmentation |
| Statistics | Bayesian inference, correlation, PCA |
| Machine Learning | Logistic Regression, XGBoost, LightGBM |
| Interpretability | SHAP |
| Experimentation | Bayesian A/B testing |
| Causal Analytics | Uplift modeling, treatment effects |
| Business | churn, CLV, campaign ROI, targeting |
| Reporting | Power BI |

---

## Key Idea

The project separates four different analytical questions:

| Question | Method |
|---|---|
| Who is likely to churn? | Churn prediction |
| Why is the model predicting churn? | SHAP |
| Does the campaign work? | Bayesian A/B testing |
| Who should receive the offer? | Uplift + expected profit |

The goal is not simply to create the most accurate churn classifier.

The goal is to build a practical retention decision system that connects customer analytics, statistical inference, model interpretation, and financial outcomes.
