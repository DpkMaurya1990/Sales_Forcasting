# src/features.py
import pandas as pd
import numpy as np

def build_features(df, oil=None, holidays=None, is_train=True):
    """
    Builds features for both training and inference.
    Must be deterministic and reusable.
    """

    df = df.copy()

    # -------------------
    # Missing values
    # -------------------
    if "onpromotion" in df.columns:
        df["onpromotion"] = df["onpromotion"].fillna(0)

    # -------------------
    # Date features
    # -------------------
    df["day_of_week"] = df["date"].dt.weekday
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month

    # -------------------
# -------------------
# Lag & rolling features (only if sales exists)
# -------------------
    if "sales" in df.columns:
        for lag in [1, 7, 14, 28]:
            df[f"sales_lag_{lag}"] = (
                df.groupby(["store_nbr", "family"])["sales"]
                  .shift(lag)
            )
            
        for window in [7, 14, 28]:
            df[f"sales_roll_mean_{window}"] = (
                df.groupby(["store_nbr", "family"])["sales"]
                  .shift(1)
                  .rolling(window)
                  .mean()
            )

    # -------------------
    # Drop rows with NaNs from lag/rolling
    # -------------------
    if is_train:
        df = df.dropna().reset_index(drop=True)

    return df
