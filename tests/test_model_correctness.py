import joblib
import pandas as pd

# Define path to our model
MODEL_DIR = "models"


def test_model_correctness():
    clf = joblib.load(f"{MODEL_DIR}/model.pkl")
    data = {
        "customer_age": 25,
        "tenure": 12,
        "account_balance": 456.78,
        "previous_campaign_number_of_contacts": 0,
        "employment_type": "student",
        "marital_status": "single",
        "education_level": "secondary",
        "credit_default": "no",
        "housing_loan": "no",
        "personal_loan": "no",
        "contact_method": "cellular",
        "previous_campaign_outcome": "unknown",
    }
    assert clf.predict(pd.DataFrame(data, index=[0])) == 0
