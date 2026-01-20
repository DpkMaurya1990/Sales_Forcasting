#I separated training and inference pipelines to ensure production consistency. inference.py guarantees 
# feature parity with training, handles categorical enforcement, business rules, and is the exact script deployed on EC2
# or behind FastAPI.

import joblib
import pandas as pd
import numpy as np
from features import build_features

MODEL_PATH = "E:/store_sales_forecasting/models/lightgbm_model.pkl"
FEATURES_PATH = "E:/store_sales_forecasting/models/features.pkl"
DATA_PATH = "E:/store_sales_forecasting/data/raw/"

def predict():
    model = joblib.load(MODEL_PATH)
    FEATURES = joblib.load(FEATURES_PATH)

    # load data
    test = pd.read_csv(DATA_PATH + "test.csv", parse_dates=["date"])
    oil = pd.read_csv(DATA_PATH + "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(DATA_PATH + "holidays_events.csv")

    # load recent history
    train_hist = pd.read_csv(
        DATA_PATH + "train.csv",
        parse_dates=["date"]
    )

    train_hist = (
        train_hist
        .sort_values("date")
        .groupby(["store_nbr", "family"])
        .tail(28)
    )

    # add placeholder + combine
    test["sales"] = np.nan
    combined = pd.concat([train_hist, test], ignore_index=True)

    # feature engineering
    combined = build_features(
        combined,
        oil=oil,
        holidays=holidays,
        is_train=False
    )

    # keep only test rows
    test = combined[combined["sales"].isna()].copy()

    X_test = test[FEATURES].copy()
    for col in X_test.select_dtypes("object").columns:
        X_test[col] = X_test[col].astype("category")

    preds = model.predict(X_test)
    preds = np.clip(preds, a_min=0, a_max=None)
    test["sales_pred"] = preds

    test[["id", "sales_pred"]].to_csv(
        "E:/store_sales_forecasting/outputs/inference_predictions.csv",
        index=False
    )

    print("Inference completed successfully.")

if __name__ == "__main__":
    predict()
