import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    return out

def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_columns(df)
    rename = {
        "customerid": "customer_id",
        "seniorcitizen": "senior_citizen",
        "monthlycharges": "monthly_charges",
        "totalcharges": "total_charges",
    }
    out = out.rename(columns=rename)
    if "total_charges" in out:
        out["total_charges"] = pd.to_numeric(out["total_charges"], errors="coerce")
        out["total_charges"] = out["total_charges"].fillna(0)
    if "churn" in out:
        out["churn_flag"] = out["churn"].map({"Yes": 1, "No": 0})
    return out

def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "tenure" in out:
        out["tenure_group"] = pd.cut(
            out["tenure"], bins=[-1, 12, 24, 48, np.inf],
            labels=["0-12", "13-24", "25-48", "49+"]
        )
    if "contract" in out:
        out["month_to_month_flag"] = (out["contract"]=="Month-to-month").astype(int)
    elif "contract_type" in out:
        out["month_to_month_flag"] = (out["contract_type"]=="Month-to-month").astype(int)

    service_cols = [c for c in [
        "phoneservice", "multiplelines", "onlinesecurity", "onlinebackup",
        "deviceprotection", "techsupport", "streamingtv", "streamingmovies"
    ] if c in out.columns]
    if service_cols:
        out["number_of_services"] = sum((out[c]=="Yes").astype(int) for c in service_cols)
    elif "num_services" in out:
        out["number_of_services"] = out["num_services"]

    if "monthly_charges" in out:
        q = out["monthly_charges"].quantile(0.75)
        out["high_value_customer"] = (out["monthly_charges"] >= q).astype(int)
    return out

def make_preprocessor(df: pd.DataFrame, target="churn_flag"):
    X = df.drop(columns=[target], errors="ignore")
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    categorical = [c for c in categorical if c != "customer_id"]
    numeric = [c for c in numeric if c != "customer_id"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return preprocessor, numeric, categorical
