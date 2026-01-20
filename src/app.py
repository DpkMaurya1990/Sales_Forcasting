from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib

from src.features import build_features

DATA_PATH = "data/raw/"
MODEL_PATH = "models/lightgbm_model.pkl"
FEATURES_PATH = "models/features.pkl"

app = FastAPI(title="Store Sales Forecast API")

# --------------------
# Load artifacts ONCE (industry practice)
# --------------------
model = joblib.load(MODEL_PATH)
FEATURES = joblib.load(FEATURES_PATH)

oil = pd.read_csv(DATA_PATH + "oil.csv", parse_dates=["date"])
holidays = pd.read_csv(DATA_PATH + "holidays_events.csv")
train_hist = pd.read_csv(DATA_PATH + "train.csv", parse_dates=["date"])

train_hist = (
    train_hist
    .sort_values("date")
    .groupby(["store_nbr", "family"])
    .tail(28)
)

# --------------------
# Health check
# --------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# --------------------
# Prediction endpoint
# --------------------
@app.post("/predict")
def predict(data: list[dict]):
    test = pd.DataFrame(data)

    # convert date string → datetime (CRITICAL for APIs)
    test["date"] = pd.to_datetime(test["date"])

    test["sales"] = np.nan

    combined = pd.concat([train_hist, test], ignore_index=True)

    combined = build_features(
        combined,
        oil=oil,
        holidays=holidays,
        is_train=False
    )

    test = combined[combined["sales"].isna()].copy()
    X_test = test[FEATURES].copy()

    for col in X_test.select_dtypes("object").columns:
        X_test[col] = X_test[col].astype("category")

    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)

    return {"predictions": preds.tolist()}