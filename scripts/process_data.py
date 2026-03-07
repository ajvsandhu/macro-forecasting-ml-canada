# Data cleaning & preprocessing — filters, merges all macro data, adds features

from pathlib import Path
import pandas as pd
import numpy as np

# Paths relative to this script
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


# Normalizes any date column to the first of the month so all datasets align
def to_month_start(df):
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp() # Ex. 2009-04-30 -> 2009-04 -> 2009-04-01
    return df

# Loads the unemployment rate from the StatCan labour force table
def load_unemployment():
    df = pd.read_csv(RAW_DIR / "statcan_labour_force.csv", low_memory=False)

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

    return to_month_start(df)


# Loads the employment rate from the same StatCan table 
def load_employment_rate():
    # Same table as unemployment, just filtering for "Employment rate" instead
    df = pd.read_csv(RAW_DIR / "statcan_labour_force.csv", low_memory=False)

    mask = (
        (df["GEO"] == "Canada") &
        (df["Labour force characteristics"] == "Employment rate") &
        (df["Gender"] == "Total - Gender") &
        (df["Age group"] == "15 years and over") &
        (df["Data type"] == "Seasonally adjusted") &
        (df["Statistics"] == "Estimate")
    )
    df = df[mask].copy()

    df["date"] = pd.to_datetime(df["REF_DATE"])
    df = df[["date", "VALUE"]].rename(columns={"VALUE": "employment_rate"})

    return to_month_start(df)


# Loads the CPI (Consumer Price Index) from StatCan
def load_cpi():
    df = pd.read_csv(RAW_DIR / "statcan_cpi.csv", low_memory=False)

    # overall price level based on how they were in 2002 (100 is the base year)
    mask = (
        (df["GEO"] == "Canada") &
        (df["Products and product groups"] == "All-items") &
        (df["UOM"] == "2002=100")
    )
    df = df[mask].copy()

    df["date"] = pd.to_datetime(df["REF_DATE"])
    df = df[["date", "VALUE"]].rename(columns={"VALUE": "cpi"})

    return to_month_start(df)


# Loads monthly GDP from StatCan
def load_gdp():
    df = pd.read_csv(RAW_DIR / "statcan_gdp.csv", low_memory=False)

    # All industries = total economy, seasonally adjusted, in chained 2017 dollars (inflation-adjusted)
    mask = (
        (df["GEO"] == "Canada") &
        (df["North American Industry Classification System (NAICS)"] == "All industries [T001]") &
        (df["Seasonal adjustment"] == "Seasonally adjusted at annual rates") &
        (df["Prices"] == "Chained (2017) dollars")
    )
    df = df[mask].copy()

    df["date"] = pd.to_datetime(df["REF_DATE"])
    df = df[["date", "VALUE"]].rename(columns={"VALUE": "gdp"})

    return to_month_start(df)


# Loads the monthly weighted-average overnight rate from BoC
def load_overnight_rate():
    df = pd.read_csv(RAW_DIR / "boc_overnight_rate.csv")

    # Convert date to datetime object
    df["date"] = pd.to_datetime(df["date"])

    return to_month_start(df)


# Loads the monthly 10-year bond yield from BoC
def load_bond_yield():
    df = pd.read_csv(RAW_DIR / "boc_bond_yield_10y.csv")

    # Convert date to datetime object
    df["date"] = pd.to_datetime(df["date"])

    return to_month_start(df)


# Loads the monthly CAD/USD exchange rate from FRED
def load_exchange_rate():
    df = pd.read_csv(RAW_DIR / "fred_exchange_rate.csv")

    # Convert date to datetime object
    df["date"] = pd.to_datetime(df["date"])

    return to_month_start(df)


# Loads the monthly WTI oil price from FRED
def load_oil_price():
    df = pd.read_csv(RAW_DIR / "fred_oil_price_wti.csv")

    # Convert date to datetime object
    df["date"] = pd.to_datetime(df["date"])

    return to_month_start(df)

# Merges all datasets into one DataFrame by month 
def merge_all(datasets):
    # Start with the first dataset
    df = datasets[0]

    # Merge each remaining dataset one by one on the date column
    for other in datasets[1:]:
        df = pd.merge(df, other, on="date", how="inner")

    # Sort by date from oldest to newest
    df = df.sort_values("date").reset_index(drop=True)

    return df

# Adds lag features, rate of change, and rolling averages to the merged DataFrame
def add_features(df):

    # Lagged overnight rate: what was the rate 3, 6, and 12 months ago?
    for lag in [3, 6, 12]:
        df[f"rate_lag_{lag}m"] = df["overnight_rate"].shift(lag)

    # How much the overnight rate changed since last month
    df["rate_change_1m"] = df["overnight_rate"].diff(1)

    # How much the overnight rate changed over the last 3 months
    df["rate_change_3m"] = df["overnight_rate"].diff(3)

    # The average overnight rate over the last 6 months
    df["rate_rolling_6m"] = df["overnight_rate"].rolling(window=6).mean()

    # Last month's unemployment rate
    df["unemp_lag_1m"] = df["unemployment_rate"].shift(1)

    # How much unemployment changed since last month
    df["unemp_change_1m"] = df["unemployment_rate"].diff(1)

    # The average unemployment rate over the last 12 months
    df["unemp_rolling_12m"] = df["unemployment_rate"].rolling(window=12).mean()

    # Month-over-month inflation (percent change in CPI)
    df["cpi_change_1m"] = df["cpi"].pct_change(1) * 100

    # Year-over-year inflation this month compared to last year month)
    df["cpi_change_12m"] = df["cpi"].pct_change(12) * 100

    # Month-over-month GDP growth
    df["gdp_change_1m"] = df["gdp"].pct_change(1) * 100

    # Year-over-year GDP growth
    df["gdp_change_12m"] = df["gdp"].pct_change(12) * 100

    # When this goes negative, recessions tend to follow
    df["yield_spread"] = df["bond_yield_10y"] - df["overnight_rate"]

    # --- Oil price features ---

    # 3-month percent change in oil price (captures energy shocks)
    df["oil_change_3m"] = df["oil_price_wti"].pct_change(3) * 100

    # --- Exchange rate features ---

    # 3-month percent change in CAD/USD (captures currency momentum)
    df["fx_change_3m"] = df["cad_usd"].pct_change(3) * 100

    return df

# Main
# Loads, merges, engineers features, and saves the final dataset
def main():
    # Create output directory
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load all 8 datasets
    print("Loading all datasets")
    unemp = load_unemployment()
    emp = load_employment_rate()
    cpi = load_cpi()
    gdp = load_gdp()
    overnight = load_overnight_rate()
    bond = load_bond_yield()
    fx = load_exchange_rate()
    oil = load_oil_price()
    print(f"  Loaded 8 datasets")

    # Merge everything into one DataFrame by month
    print("Merging all datasets by month")
    df = merge_all([unemp, emp, cpi, gdp, overnight, bond, fx, oil])
    print(f"  {len(df)} months ({df['date'].min().date()} to {df['date'].max().date()})")

    # Add engineered features (lags, changes, rolling averages, yield spread)
    print("Adding features")
    df = add_features(df)

    # Drop rows where lag features created NaNs 
    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"  Dropped {rows_before - len(df)} rows with NaN from lag features")

    # Save the final dataset
    out_path = PROCESSED_DIR / "merged_monthly.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {out_path} ({len(df)} rows, {len(df.columns)} columns)")

if __name__ == "__main__":
    main()
