# Step 4: Exploratory Data Analysis — understand the data before modeling

from typing import Any


from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") # renders plots to file instead of screen
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller # stationarity tests

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "eda"

# The 8 raw variables (before feature engineering)
RAW_COLS = [
    "unemployment_rate", "employment_rate", "cpi", "gdp",
    "overnight_rate", "bond_yield_10y", "cad_usd", "oil_price_wti"
]


# Summary statistics 

# Prints summary stats for all columns and saves to CSV
def summary_stats(df):
    print("\n=== Summary Statistics ===\n")
    stats = df[RAW_COLS].describe().T
    stats["missing"] = df[RAW_COLS].isnull().sum()
    print(stats.to_string()) # print summary stats to console

    stats.to_csv(OUTPUT_DIR / "summary_stats.csv")
    print(f"\n  Saved to {OUTPUT_DIR / 'summary_stats.csv'}") 


# Correlation matrix 

# Creates a heatmap showing how every variable correlates with every other variable
def correlation_matrix(df):

    # Correlate all features against unemployment_rate, sorted by strength
    corr = df.drop(columns=["date"]).corr()
    unemp_corr = corr["unemployment_rate"].drop("unemployment_rate").sort_values(key=abs, ascending=False)

    # Save the full correlation rankings to CSV
    unemp_corr.to_csv(OUTPUT_DIR / "correlation_vs_unemployment.csv")

    # Human-readable names for the heatmap labels
    label_names = {
        "unemployment_rate": "Unemployment Rate",
        "employment_rate": "Employment Rate",
        "cpi": "Consumer Price Index",
        "gdp": "Gross Domestic Product",
        "overnight_rate": "BoC Overnight Rate",
        "bond_yield_10y": "10-Year Bond Yield",
        "cad_usd": "CAD/USD Exchange Rate",
        "oil_price_wti": "WTI Oil Price",
    }

    # Rename columns for the heatmap so it shows readable names
    corr_matrix = df[RAW_COLS].rename(columns=label_names).corr()

    # Heatmap of raw variables only (full matrix is too big to read)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        square=True, linewidths=0.5, ax=ax
    )
    ax.set_title("Correlation Between Variables", fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=150)
    plt.close()

# Distribution plots 

# Histogram for each raw variable to check for skew and outliers
def distribution_plots(df):

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("How Often Each Value Occurs (2010–2024, 177 months)", fontsize=14, fontweight="bold")

    # Units for each variable so the x-axis labels are clear
    units = {
        "unemployment_rate": "Value (%)",
        "employment_rate": "Value (%)",
        "cpi": "Value (% above 2002 prices)",
        "gdp": "Value ($ millions)",
        "overnight_rate": "Value (%)",
        "bond_yield_10y": "Value (%)",
        "cad_usd": "Value (CAD per 1 USD)",
        "oil_price_wti": "Value (USD/barrel)",
    }

    # Human-readable titles for each variable
    titles = {
        "unemployment_rate": "Unemployment Rate",
        "employment_rate": "Employment Rate",
        "cpi": "Consumer Price Index",
        "gdp": "Gross Domestic Product",
        "overnight_rate": "BoC Overnight Rate",
        "bond_yield_10y": "10-Year Bond Yield",
        "cad_usd": "CAD/USD Exchange Rate",
        "oil_price_wti": "WTI Oil Price",
    }

    for ax, col in zip(axes.flat, RAW_COLS):
        # For CPI, convert index values to % above 2002 (e.g. 120 -> 20%)
        # Uses Freedman-Diaconis rule for optimal bin size
        if col == "cpi":
            plot_data = df[col] - 100
            ax.hist(plot_data, bins="fd", color="#3498db", edgecolor="white", alpha=0.8)
            ax.axvline(plot_data.mean(), color="red", linestyle="--", label="mean")
        else:
            ax.hist(df[col], bins="fd", color="#3498db", edgecolor="white", alpha=0.8)
            ax.axvline(df[col].mean(), color="red", linestyle="--", label="mean")

        ax.set_title(titles[col])
        ax.set_xlabel(units[col])
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distributions.png", dpi=150)
    plt.close()
    print(f"  Saved to {OUTPUT_DIR / 'distributions.png'}")


# Stationarity tests (Augmented Dickey-Fuller) 

# Tests whether each variable has a trend or fluctuates around a constant mean
def stationarity_tests(df):
    # Test stationarity of each variable
    stationarity_results = []
    for col in RAW_COLS:
        series = df[col].dropna()
        adf_stat, p_value, _, _, _, _ = adfuller(series, autolag="AIC")
        stationarity_results.append({
            "variable": col,
            "adf_stat": adf_stat,
            "p_value": p_value,
            "stationary": "Yes" if p_value < 0.05 else "No"
        })

    pd.DataFrame(stationarity_results).to_csv(OUTPUT_DIR / "stationarity_tests.csv", index=False)


# How many months does it take for an overnight rate change to show up in unemployment?
# Tests lags from 0 to 36 months to capture the full cycle
def lagged_correlation(df):

    max_lag = 36
    lags = range(0, max_lag + 1)
    correlations = []

    for lag in lags:
        # Shift overnight rate back by 'lag' months and correlate with unemployment
        corr = df["overnight_rate"].shift(lag).corr(df["unemployment_rate"])
        correlations.append(corr)

    # Find the lag with the strongest positive correlation (the delayed effect)
    positive_corrs = [(l, c) for l, c in zip(lags, correlations) if c > 0]
    if positive_corrs:
        peak_lag, peak_corr = max(positive_corrs, key=lambda x: x[1])
    else:
        peak_lag, peak_corr = 0, 0

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(lags, correlations, color="#3498db", edgecolor="white")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("Correlation with Unemployment Rate")
    ax.set_title("How Long Does It Take for a Rate Change to Affect Unemployment?")
    ax.set_xticks(range(0, max_lag + 1, 3))

    # Mark the peak positive lag (the delayed effect)
    ax.bar(peak_lag, peak_corr, color="#e74c3c", edgecolor="white")
    ax.annotate(f"Peak effect: {peak_lag}mo (r={peak_corr:+.3f})",
                xy=(peak_lag, peak_corr), xytext=(peak_lag + 3, peak_corr + 0.05),
                arrowprops=dict(arrowstyle="->"), fontsize=10)

    # Mark the crossover point where correlation flips from negative to positive
    ax.annotate("Correlation flips\n(effect kicks in)",
                xy=(11, 0), xytext=(14, -0.25),
                arrowprops=dict(arrowstyle="->"), fontsize=9, ha="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lagged_correlation.png", dpi=150)
    plt.close()
    print(f"  Saved to {OUTPUT_DIR / 'lagged_correlation.png'}")

    # Also do it for all predictors
    predictors = ["overnight_rate", "cpi", "gdp", "bond_yield_10y", "cad_usd", "oil_price_wti", "employment_rate"]
    for pred in predictors:
        corrs = [df[pred].shift(l).corr(df["unemployment_rate"]) for l in lags]
        bl = lags[np.argmax(np.abs(corrs))]
        bc = corrs[bl]
        print(f"    {pred:25s} lag={bl:2d} months  r={bc:+.3f}")


# Time series plots with unemployment overlay

# Plots each predictor against unemployment to visually check relationships
def overlay_plots(df):

    predictors = ["overnight_rate", "cpi_change_12m", "gdp_change_12m",
                   "bond_yield_10y", "yield_spread", "oil_price_wti", "cad_usd"]

    # Human-readable titles for each predictor
    pred_titles = {
        "overnight_rate": "BoC Overnight Rate",
        "cpi_change_12m": "Year-over-Year Inflation (%)",
        "gdp_change_12m": "Year-over-Year GDP Growth (%)",
        "bond_yield_10y": "10-Year Bond Yield",
        "yield_spread": "Yield Spread (10yr - Overnight)",
        "oil_price_wti": "WTI Oil Price",
        "cad_usd": "CAD/USD Exchange Rate",
    }

    # 4 rows x 2 columns grid (7 predictors + 1 empty cell)
    fig, axes = plt.subplots(4, 2, figsize=(18, 16))
    fig.suptitle("Each Predictor (blue) vs Unemployment Rate (red)", fontsize=15, fontweight="bold")

    for i, pred in enumerate(predictors):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        # Left axis: predictor
        color1 = "#3498db"
        ax.plot(df["date"], df[pred], color=color1, linewidth=1.5)
        ax.set_ylabel(pred_titles[pred], color=color1, fontsize=9)
        ax.tick_params(axis="y", labelcolor=color1)

        # Right axis: unemployment
        ax2 = ax.twinx()
        ax2.plot(df["date"], df["unemployment_rate"], color="#e74c3c", linewidth=1.5, alpha=0.7)
        ax2.set_ylabel("Unemployment %", color="#e74c3c", fontsize=9)

        ax.set_title(pred_titles[pred], fontsize=11)
        ax.set_xlabel("Time (years)")
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis="x", rotation=30)

    # Hide the empty cell (bottom right, since 7 predictors in 4x2 = 1 leftover)
    axes[3, 1].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "overlay_plots.png", dpi=150, bbox_inches="tight")
    plt.close()

# Main 

# Runs all EDA steps and saves outputs
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the processed dataset
    df = pd.read_csv(DATA_DIR / "merged_monthly.csv")
    df["date"] = pd.to_datetime(df["date"])
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Run all analyses
    summary_stats(df)
    correlation_matrix(df)
    distribution_plots(df)
    stationarity_tests(df)
    lagged_correlation(df)
    overlay_plots(df)

    print("\n=== EDA Complete ===")
    print(f"All outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
