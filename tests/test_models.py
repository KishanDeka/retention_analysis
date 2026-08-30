import pandas as pd
from src.models import build_logistic_model

def test_build_logistic_model():
    df = pd.DataFrame({
        "tenure":[1,2,40,50,5,60],
        "monthly_charges":[100,90,40,30,95,35],
        "contract_type":["Month-to-month","Month-to-month","Two year","Two year","Month-to-month","One year"],
        "churn_flag":[1,1,0,0,1,0]
    })
    model = build_logistic_model(df)
    X = df.drop(columns="churn_flag")
    y = df["churn_flag"]
    model.fit(X,y)
    assert len(model.predict_proba(X)) == len(df)
