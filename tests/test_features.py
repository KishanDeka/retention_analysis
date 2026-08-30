import pandas as pd
from src.features import add_business_features

def test_add_business_features():
    df = pd.DataFrame({
        "tenure":[5,50],
        "monthly_charges":[100,40],
        "contract_type":["Month-to-month","Two year"],
        "num_services":[2,5]
    })
    out = add_business_features(df)
    assert "tenure_group" in out
    assert "month_to_month_flag" in out
    assert "high_value_customer" in out
