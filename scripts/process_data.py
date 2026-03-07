# Data cleaning & preprocessing — filters raw StatCan + BoC data, merges by month, adds features

from pathlib import Path
import pandas as pd
import numpy as np

# Paths relative to this script
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


# This function loads the unemployment rate data from the raw data folder and filters it to the necessary columns and rows
def load_unemployment():
    # Read the full 5.4M row table
    df = pd.read_csv(RAW_DIR / "statcan_labour_force_14-10-0287-01.csv", low_memory=False)

    # Filter to exactly what we need (huge table with many different range of rows)
    mask = ( 
        (df["GEO"] == "Canada") & # for the global unemployment rate
        (df["Labour force characteristics"] == "Unemployment rate") & # get metric for unemployment rate
        (df["Gender"] == "Total - Gender") & # get metric for both genders
        (df["Age group"] == "15 years and over") & # get metric for age group of 15 years and over
        (df["Data type"] == "Seasonally adjusted") & # get metric for seasonally adjusted data
        (df["Statistics"] == "Estimate") # get metric for estimate
    )
    df = df[mask].copy() # checks every row that all 6 conditions are true and if so gets put in a new df

    df["date"] = pd.to_datetime(df["REF_DATE"]) # converts the date column to a datetime object
    df = df[["date", "VALUE"]].rename(columns={"VALUE": "unemployment_rate"}) # keeps only the date and unemployment rate columns

    # Sort by date from oldest to youngest
    df = df.sort_values("date").reset_index(drop=True)

    return df


# This function loads the monthly weighted-average overnight rate from the raw data folder
def load_overnight_rate():

    df = pd.read_csv(RAW_DIR / "boc_overnight_rate_V39079.csv")

    # Convert date to datetime object so python can sort it
    df["date"] = pd.to_datetime(df["date"])

    return df


# This function aligns both datasets to the same monthly period and merges them into one DataFrame
def align_and_merge(unemp, rate):

    # Normalize both date columns to the first of the month so they match
    unemp["date"] = unemp["date"].dt.to_period("M").dt.to_timestamp() # Ex. 2009-04-01 -> 2009-04 
    rate["date"] = rate["date"].dt.to_period("M").dt.to_timestamp() # Ex. 2009-04-01 -> 2009-04

    # Merge on the month
    df = pd.merge(unemp, rate, on="date", how="inner")

    # Sort by date from oldest to youngest
    df = df.sort_values("date").reset_index(drop=True)

    return df


# This function adds lag features, rate of change, and rolling averages to the merged DataFrame
def add_features(df):

    # --- Overnight rate features ---

    # Lagged overnight rate, the rate 3, 6, and 12 months ago?
    for lag in [3, 6, 12]:
        df[f"rate_lag_{lag}m"] = df["overnight_rate_wavg"].shift(lag)

    # How much the overnight rate changed since last month
    df["rate_change_1m"] = df["overnight_rate_wavg"].diff(1)

    # How much the overnight rate changed over the last 3 months
    df["rate_change_3m"] = df["overnight_rate_wavg"].diff(3)

    # The average overnight rate over the last 6 months
    df["rate_rolling_6m"] = df["overnight_rate_wavg"].rolling(window=6).mean()

    # --- Unemployment features ---

    # Last month's unemployment rate.
    df["unemp_lag_1m"] = df["unemployment_rate"].shift(1)

    # How much the unemployment rate changed since last month
    df["unemp_change_1m"] = df["unemployment_rate"].diff(1)

    # The average unemployment rate over the last 12 months
    df["unemp_rolling_12m"] = df["unemployment_rate"].rolling(window=12).mean()

    return df


def main():
    # Create output directory
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw data
    print("Loading unemployment rate")
    unemp = load_unemployment()
    print(f"  {len(unemp)} months of unemployment data")

    print("Loading overnight rate")
    rate = load_overnight_rate()
    print(f"  {len(rate)} months of overnight rate data")

    # Merge
    print("Merging and aligning dates")
    df = align_and_merge(unemp, rate)
    print(f"  {len(df)} months in merged dataset ({df['date'].min().date()} to {df['date'].max().date()})")

    # Feature engineering
    print("Adding features (lags, changes, rolling averages)")
    df = add_features(df)

    # Drop rows where lag features created NaNs (first 12 months)
    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"  Dropped {rows_before - len(df)} rows with NaN from lag features")

    # Save
    out_path = PROCESSED_DIR / "merged_monthly.csv"
    df.to_csv(out_path, index=False)
    print(f"  -> {out_path} ({len(df)} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
