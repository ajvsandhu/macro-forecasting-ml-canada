# Data loading and config for modeling pipeline

from pathlib import Path
import pandas as pd
import numpy as np

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "modeling"

# The column we're trying to predict
TARGET = "unemployment_rate"

# Features that must be removed from the model before training because they
# contain current-month unemployment information (this what we are training the model to predict)
# called data leakage
# both of these features contain the current month's unemployment rate
LEAK_COLS = ["unemp_change_1m", "unemp_rolling_12m"]

# Chronological split (no shuffling — time series must stay ordered)
# first 80% for training, last 20% for testing (80-20 rule)
TRAIN_RATIO = 0.80

# Variables used in the VAR model
VAR_COLS = ["unemployment_rate", "employment_rate", "overnight_rate", "bond_yield_10y"]

# Data loading & train/test split

# Loads processed data and splits chronologically into train/test
def load_and_split():
    # Read the merged monthly dataset from process_data.py
    df = pd.read_csv(DATA_DIR / "merged_monthly.csv")
    df["date"] = pd.to_datetime(df["date"])  # parse date strings into datetime objects

    # Calculate where to split (80% of rows = training set)
    split_idx = int(len(df) * TRAIN_RATIO)

    # Split chronologically (no shuffling — time series must stay ordered)
    train_df = df.iloc[:split_idx].copy()  # first 80% of months
    test_df = df.iloc[split_idx:].copy()  # last 20% of months

    # Build feature list: everything except date, target, and leaky columns
    feature_cols = [c for c in df.columns if c not in ["date", TARGET] + LEAK_COLS]

    # Separate features (X) and target/prediction variable (y) for the ML models

    # stores and returns the features and target/prediction variable for the training and test sets
    # shape of array: (rows, columns): (n_train, n_features)
    X_train = train_df[feature_cols].values  # (n_train, n_features)
    X_test = test_df[feature_cols].values  # (n_test, n_features)
    y_train = train_df[TARGET].values  # (n_train,)
    y_test = test_df[TARGET].values  # (n_test,)

    # Store dates for plotting later
    dates_train = train_df["date"].values
    dates_test = test_df["date"].values

    print(f"\n=== Train/Test Split ===")
    print(f"  Train: {len(train_df)} months ({train_df['date'].iloc[0].date()} to {train_df['date'].iloc[-1].date()})")
    print(f"  Test:  {len(test_df)} months ({test_df['date'].iloc[0].date()} to {test_df['date'].iloc[-1].date()})")
    print(f"  Features: {len(feature_cols)} columns (after removing date, target, and leaky columns)")

    return (train_df, test_df, X_train, X_test, y_train, y_test,
            dates_train, dates_test, feature_cols)
