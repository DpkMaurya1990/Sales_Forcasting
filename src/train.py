# This is to prevents accidental execution on import
#I separated experimentation from production. train.py is a deterministic, reproducible training pipeline that can be triggered
# via CLI or CI/CD, ensuring consistent retraining and artifact versioning.
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from features import build_features

DATA_PATH = "E:/store_sales_forecasting/data/raw/"
MODEL_PATH = "E:/store_sales_forecasting/models/"

def train():
    # 1. load raw data
    train_df = pd.read_csv(DATA_PATH + "train.csv", parse_dates=["date"])
    stores = pd.read_csv(DATA_PATH + "stores.csv")
    oil = pd.read_csv(DATA_PATH + "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(DATA_PATH + "holidays_events.csv")
    
    train_df = build_features(
    train_df,
    oil=oil,
    holidays=holidays,
    is_train=True
)

    # 3. prepare training matrix
    y = train_df["sales"]
    X = train_df.drop(columns=["sales", "date", "id"])

    # categorical handling
    for col in X.select_dtypes("object").columns:
        X[col] = X[col].astype("category")

    # 4. train model
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        objective="regression"
    )
    model.fit(X, y)

    # 5. save artifacts
    FEATURES = X.columns.tolist()

    joblib.dump(model, MODEL_PATH + "lightgbm_model.pkl")
    joblib.dump(FEATURES, MODEL_PATH + "features.pkl")

    print("Training completed successfully.")

if __name__ == "__main__":
    train()

