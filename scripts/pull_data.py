# Step 2: Data acquisition — pulls all macro data needed for the project

from pathlib import Path
import zipfile
import pandas as pd
import requests

# Build the path to data/raw/ relative to wherever this script lives
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Date range for the data we want to pull
START_DATE = "2005-01-01"
END_DATE = "2024-12-31"


# This function pulls the StatCan labour force table and returns it as a DataFrame
def pull_statcan_unemployment(data_dir):

    # download_tables downloads the full table as a zip from StatCan
    from stats_can.sc import download_tables
    from stats_can.helpers import parse_tables

    # Convert the human-readable table ID to the numeric format API expects
    table = parse_tables("14-10-0287-01")[0]

    # Build the expected zip file path, e.g. data/raw/14100287-eng.zip
    zip_path = data_dir / f"{table}-eng.zip"

    # Only download if we don't already have the zip
    if not zip_path.is_file():
        download_tables([table], data_dir)

    # The CSV inside the zip is named like 14100287.csv
    csv_name = f"{table}.csv"

    # Open the zip archive
    with zipfile.ZipFile(zip_path) as z:
        with z.open(csv_name) as f:
            # Read the CSV into a pandas DataFrame
            df = pd.read_csv(f, low_memory=False)

    # Return the full table as a DataFrame (datetime conversion happens in process_data.py)
    return df

# This function pulls any daily series from the Bank of Canada Valet API and returns it as a monthly DataFrame
def pull_boc_series(series_id, col_name):

    # Build the Valet API URL for this series
    url = f"https://www.bankofcanada.ca/valet/observations/{series_id}/json"

    # Pass our date range as query parameters
    params = {"start_date": START_DATE, "end_date": END_DATE}

    # Send the HTTP GET request (wait up to 30 seconds)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    # Parse the JSON response
    data = r.json()

    # Extract daily observations
    rows = []
    for obs in data.get("observations", []):
        val = obs.get(series_id, {}).get("v")
        if val is not None:
            rows.append({"date": obs["d"], col_name: float(val)})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    # Forward-fill weekends/holidays, then take monthly weighted average
    df = df.set_index("date").resample("D").ffill()
    df = df.resample("ME").mean().reset_index()

    return df


# This function pulls StatCan CPI data (Table 18-10-0004-01) and returns it as a DataFrame
def pull_statcan_cpi(data_dir):

    from stats_can.sc import download_tables
    from stats_can.helpers import parse_tables

    table = parse_tables("18-10-0004-01")[0]
    zip_path = data_dir / f"{table}-eng.zip"

    if not zip_path.is_file():
        download_tables([table], data_dir)

    csv_name = f"{table}.csv"
    with zipfile.ZipFile(zip_path) as z:
        with z.open(csv_name) as f:
            df = pd.read_csv(f, low_memory=False)

    return df


# This function pulls StatCan GDP data (Table 36-10-0434-01) and returns it as a DataFrame
def pull_statcan_gdp(data_dir):

    from stats_can.sc import download_tables
    from stats_can.helpers import parse_tables

    table = parse_tables("36-10-0434-01")[0]
    zip_path = data_dir / f"{table}-eng.zip"

    if not zip_path.is_file():
        download_tables([table], data_dir)

    csv_name = f"{table}.csv"
    with zipfile.ZipFile(zip_path) as z:
        with z.open(csv_name) as f:
            df = pd.read_csv(f, low_memory=False)

    return df


# This function pulls WTI oil prices from FRED (free, no API key needed)
def pull_fred_oil_price():

    # FRED provides a direct CSV download for any series (free, no API key)
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=MCOILWTICO&cosd={START_DATE}&coed={END_DATE}"
    df = pd.read_csv(url)

    # FRED returns columns: observation_date, MCOILWTICO
    df = df.rename(columns={"observation_date": "date", "MCOILWTICO": "oil_price_wti"})
    df["date"] = pd.to_datetime(df["date"])

    return df


# This is the main function that runs all pulls and saves the results
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- Bank of Canada (Valet API) ---

    print("Pulling BoC overnight rate (V39079)")
    overnight = pull_boc_series("V39079", "overnight_rate")
    overnight.to_csv(DATA_DIR / "boc_overnight_rate.csv", index=False)
    print(f"  {len(overnight)} rows")

    print("Pulling BoC exchange rate CAD/USD (FXCADUSD)")
    fx = pull_boc_series("FXCADUSD", "cad_usd")
    fx.to_csv(DATA_DIR / "boc_exchange_rate.csv", index=False)
    print(f"  {len(fx)} rows")

    print("Pulling BoC 10-year bond yield (BD.CDN.10YR.DQ.YLD)")
    bond = pull_boc_series("BD.CDN.10YR.DQ.YLD", "bond_yield_10y")
    bond.to_csv(DATA_DIR / "boc_bond_yield_10y.csv", index=False)
    print(f"  {len(bond)} rows")

    # --- FRED ---

    print("Pulling FRED WTI oil price (MCOILWTICO)")
    oil = pull_fred_oil_price()
    oil.to_csv(DATA_DIR / "fred_oil_price_wti.csv", index=False)
    print(f"  {len(oil)} rows")

    # --- Statistics Canada ---

    print("Pulling StatCan labour force (14-10-0287-01)")
    labour = pull_statcan_unemployment(DATA_DIR)
    labour.to_csv(DATA_DIR / "statcan_labour_force.csv", index=False)
    print(f"  {len(labour)} rows")

    print("Pulling StatCan CPI (18-10-0004-01)")
    cpi = pull_statcan_cpi(DATA_DIR)
    cpi.to_csv(DATA_DIR / "statcan_cpi.csv", index=False)
    print(f"  {len(cpi)} rows")

    print("Pulling StatCan GDP (36-10-0434-01)")
    gdp = pull_statcan_gdp(DATA_DIR)
    gdp.to_csv(DATA_DIR / "statcan_gdp.csv", index=False)
    print(f"  {len(gdp)} rows")

    print("Done - all data saved to data/raw/")


# This runs main() only when the script is executed directly (not when imported)
if __name__ == "__main__":
    main()
