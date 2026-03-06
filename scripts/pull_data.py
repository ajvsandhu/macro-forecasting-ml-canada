"""
Step 2: Data acquisition.
Pulls Canadian unemployment (StatCan) and overnight rate (Bank of Canada), 2005–2024.
Outputs: CSV files in data/raw/
"""

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
    """Labour force characteristics, monthly, SA (Table 14-10-0287-01)."""

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

    # Convert the REF_DATE column from strings to proper datetime objects Ex. 2005-01 -> 2005-01-01(easier to work with)
    df["REF_DATE"] = pd.to_datetime(df["REF_DATE"], errors="coerce")

    # Return the full table as a DataFrame
    return df

# This function pulls the Bank of Canada overnight rate (Daily -> monthly average) and returns it as a DataFrame
def pull_boc_overnight_rate():
    """Bank of Canada target for the overnight rate (V39079)"""

    # The Valet API endpoint for series V39079 in JSON format
    url = "https://www.bankofcanada.ca/valet/observations/V39079/json"

    # Pass our date range as query parameters (?start_date=...&end_date=...)
    params = {"start_date": START_DATE, "end_date": END_DATE}

    # Send the HTTP GET request to the Bank of Canada API (wait up to 30 seconds)
    r = requests.get(url, params=params, timeout=30)

    # If the request failed (e.g. 404, 500), raise an error immediately
    r.raise_for_status()

    # Parse the JSON response into a Python dictionary for readability
    data = r.json()

    # The series ID — used as a key in the JSON response
    series_id = "V39079"

    rows = []

    # Get each daily observation 
    for obs in data.get("observations", []):
        # The value is nested: obs["V39079"]["v"], Ex. {"V39079": {"v": "5.00"}} -> val ="5.00"
        val = obs.get(series_id, {}).get("v")

        # Only add rows where we actually got a value
        if val is not None:
            # Store the date and rate (converted from string to float)
            rows.append({"date": obs["d"], "overnight_rate": float(val)})

    # Convert our list of rows into a pandas DataFrame
    df = pd.DataFrame(rows)

    # Convert the date column from strings to datetime objects Ex. 2005-01 -> 2005-01-01(easier to work with)
    df["date"] = pd.to_datetime(df["date"])

    # Forward-fill rates to every calendar day so weekends/holidays carry the last known rate (rates cant change on weekends/holidays)
    df = df.set_index("date").resample("D").ffill()

    # Weighted average: every day counts equally, so a rate in effect for 20 days weighs more  
    # than one for 10 (accounts for mid month overnight rate changes)
    df = df.resample("ME").mean().reset_index()

    # Rename to make it clear this is a weighted monthly average
    df = df.rename(columns={"overnight_rate": "overnight_rate_wavg"})

    # Return the monthly weighted-average overnight rate DataFrame
    return df


# This is the main function that runs both pulls and saves the results
def main():
    # Create the data/raw/ folder if it doesn't exist yet
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- Bank of Canada ---
    print("Pulling Bank of Canada overnight rate (V39079)...")

    # Call the function to pull overnight rate data
    overnight = pull_boc_overnight_rate()

    # Build the output file path
    out_overnight = DATA_DIR / "boc_overnight_rate_V39079.csv"

    # Save the DataFrame to CSV (no extra index column)
    overnight.to_csv(out_overnight, index=False)

    # Print confirmation with file path and row count
    print(f"  -> {out_overnight} ({len(overnight)} rows)")

    # --- Statistics Canada ---
    print("Pulling Statistics Canada labour force (14-10-0287-01)...")

    # Call the function to pull labour force data
    labour = pull_statcan_unemployment(DATA_DIR)

    # Build the output file path
    out_labour = DATA_DIR / "statcan_labour_force_14-10-0287-01.csv"

    # Save the DataFrame to CSV
    labour.to_csv(out_labour, index=False)

    # Print confirmation
    print(f"  -> {out_labour} ({len(labour)} rows)")

    print("Done.")


# This runs main() only when the script is executed directly (not when imported)
if __name__ == "__main__":
    main()
