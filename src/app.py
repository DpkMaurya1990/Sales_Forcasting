from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import boto3

from src.features import build_features


# ---------- PATHS & CONFIG ----------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw"
# ---------- S3 CONFIG ----------
S3_BUCKET = "store-sales-forecast-models-deepak"
MODEL_KEY = "v1/lightgbm_model.pkl"
FEATURES_KEY = "v1/features.pkl"

LOCAL_MODEL_DIR = BASE_DIR / "models"
LOCAL_MODEL_PATH = LOCAL_MODEL_DIR / "lightgbm_model.pkl"
LOCAL_FEATURES_PATH = LOCAL_MODEL_DIR / "features.pkl"

# ensure local models directory exists
LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------- DOWNLOAD FROM S3 ----------
s3 = boto3.client("s3")

if not LOCAL_MODEL_PATH.exists():
    s3.download_file(S3_BUCKET, MODEL_KEY, str(LOCAL_MODEL_PATH))

if not LOCAL_FEATURES_PATH.exists():
    s3.download_file(S3_BUCKET, FEATURES_KEY, str(LOCAL_FEATURES_PATH))

# ---------- LOAD ARTIFACTS ----------
model = joblib.load(LOCAL_MODEL_PATH)
FEATURES = joblib.load(LOCAL_FEATURES_PATH)


oil = pd.read_csv(DATA_PATH / "oil.csv", parse_dates=["date"])
holidays = pd.read_csv(DATA_PATH / "holidays_events.csv")
train_hist = pd.read_csv(DATA_PATH / "train.csv", parse_dates=["date"])

train_hist = (
    train_hist
    .sort_values("date")
    .groupby(["store_nbr", "family"])
    .tail(28)
)
app = FastAPI(title="Store Sales Forecast API")
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