from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
customer_path = ROOT/"data/processed/demo_customers.csv"
experiment_path = ROOT/"data/processed/demo_retention_experiment.csv"

if not customer_path.exists() or not experiment_path.exists():
    exec((ROOT/"data/generate_demo_data.py").read_text())
    main()

customers = pd.read_csv(customer_path)
exp = pd.read_csv(experiment_path)

customers["tenure_group"] = pd.cut(
    customers["tenure"], [-1,12,24,48,999],
    labels=["0-12","13-24","25-48","49+"]
)

customers["risk_score"] = (
    3*(customers["contract_type"]=="Month-to-month").astype(int)
    + 2*(customers["tenure"]<12).astype(int)
    + 1*(customers["tech_support"]=="No").astype(int)
    + 1*(customers["monthly_charges"]>=85).astype(int)
    + 1*(customers["payment_method"]=="Electronic check").astype(int)
)
customers["risk_segment"] = pd.cut(
    customers["risk_score"], [-1,2,5,99], labels=["Low","Medium","High"]
)

out = customers.merge(
    exp[["customer_id","treatment","churn_after_campaign","campaign_cost"]],
    on="customer_id", how="left"
)
out.to_csv(ROOT/"data/processed/powerbi_customer_retention.csv", index=False)
print("Wrote data/processed/powerbi_customer_retention.csv")
