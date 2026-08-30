# Customer Retention Decision System

End-to-end portfolio project for customer churn analysis, Bayesian A/B testing,
uplift estimation, financial decision optimization, SQL analytics, and Power BI reporting.

## Business problem
A telecom company wants to reduce churn without wasting retention budget.
The project answers four questions:

1. Who is likely to churn?
2. Does a retention offer reduce churn overall?
3. Which customers are most responsive to the intervention?
4. Which customers should be targeted to maximize expected profit?

## Core stack
- SQL
- Python / pandas / scikit-learn
- XGBoost / LightGBM (optional)
- Bayesian statistics
- EconML / CausalML (optional)
- Power BI
- pytest

## Suggested datasets
### Core churn analysis
IBM Telco Customer Churn dataset.
Expected raw filename:
`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

### Causal/uplift extension
Criteo Uplift Prediction Dataset.
Use it for the causal-analysis notebook if you want a real randomized treatment dataset.

## Project structure
```text
telco-retention-analytics/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── sql/
├── notebooks/
├── src/
├── powerbi/
├── reports/
└── tests/
```

## Analytical workflow
Raw data
→ SQL cleaning and KPI extraction
→ feature engineering
→ churn-driver analysis
→ PCA
→ churn-risk modeling
→ Bayesian A/B test
→ uplift / heterogeneous treatment effect analysis
→ expected-profit targeting
→ Power BI dashboard

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python data/generate_demo_data.py
pytest
```

Then open the notebooks in order.

## Important methodological note
The IBM Telco dataset does not contain a randomized retention experiment.
The demo experiment in this repository is simulated and is explicitly labeled as such.
For real uplift modeling, use a randomized dataset such as Criteo Uplift.

## Power BI pages
1. Executive overview
2. Churn drivers and customer segments
3. Retention experiment and campaign ROI

## CV-ready project summary
Built an end-to-end customer-retention analytics system using SQL, Python and Power BI,
combining churn-risk modeling, Bayesian A/B testing, uplift estimation and expected-profit
optimization to identify financially optimal retention targets.
