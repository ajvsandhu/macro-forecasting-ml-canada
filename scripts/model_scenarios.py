# Scenario simulation — "what if" BoC rate scenarios

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_data import OUTPUT_DIR

# Scenario simulation

# Simulates 12-month forward trajectories under different BoC rate scenarios
# uses the Ridge model to predict unemployment over 12 months for each scenario.
def simulate_scenarios(model, X_test, feature_cols):
    print("\n=== Scenario Simulation ===")
    print("  Simulating 12-month trajectories under different BoC rate scenarios\n")

    # Start from the most recent test observation as our "today"
    baseline = X_test[-1:].copy()  # shape: (1, n_features)

    # Locate indices of all rate-related features we'll modify each month
    rate_idx = feature_cols.index("overnight_rate")
    lag3_idx = feature_cols.index("rate_lag_3m")
    lag6_idx = feature_cols.index("rate_lag_6m")
    lag12_idx = feature_cols.index("rate_lag_12m")
    chg1_idx = feature_cols.index("rate_change_1m")
    chg3_idx = feature_cols.index("rate_change_3m")
    roll6_idx = feature_cols.index("rate_rolling_6m") # 6-month rolling average of the overnight rate
    spread_idx = feature_cols.index("yield_spread") # yield spread between 10-year and 3-month bonds

    # Current values before the shock, the overnight rate, the 6-month rolling average of the overnight rate, and the yield spread
    current_rate = baseline[0, rate_idx]
    current_roll6 = baseline[0, roll6_idx]
    current_spread = baseline[0, spread_idx]

    # Five scenarios: 2 cuts, status quo, 2 hikes (in percentage points) how much the rate could change (linear)
    scenarios = [
        ("Rate cut -200bp",  -2.0), # 2 percentage points cut
        ("Rate cut -100bp",  -1.0), # 1 percentage point cut
        ("Status quo",        0.0), # no change
        ("Rate hike +100bp", +1.0), # 1 percentage point hike
        ("Rate hike +200bp", +2.0), # 2 percentage points hike
    ]

    horizon = 12  # simulate 12 months forward
    all_trajectories = {}  # store each scenario's trajectory for plotting
    results = []

    # For each scenario, simulate the unemployment rate over 12 months
    for name, delta in scenarios:
        scenario = baseline.copy()  # start from current conditions
        new_rate = current_rate + delta  # the new overnight rate

        # Set the overnight rate to its new level
        scenario[0, rate_idx] = new_rate

        trajectory = []  # will hold one predicted unemployment per month
        for month in range(horizon):
            # Gradually propagate rate to lagged features as months pass
            if month >= 3:
                scenario[0, lag3_idx] = new_rate  # 3 months in: lag now reflects new rate
            else:
                scenario[0, lag3_idx] = current_rate + delta * (month / 3)  # linear ramp

            if month >= 6:
                scenario[0, lag6_idx] = new_rate  # 6 months in: lag reflects new rate
            else:
                scenario[0, lag6_idx] = current_rate + delta * (month / 6)  # linear ramp

            # 12-month lag stays at old rate for the entire 12-month simulation
            scenario[0, lag12_idx] = current_rate

            # Rolling 6-month average gradually adjusts toward the new rate
            blend = min((month + 1) / 6, 1.0)  # fraction of 6-month window at new rate
            scenario[0, roll6_idx] = current_roll6 + delta * blend

            # Yield spread narrows as short rates rise (long rates move less)
            scenario[0, spread_idx] = current_spread - delta * blend * 0.5

            # Rate change is the full shock in month 0, then 0 (rate stays flat)
            if month == 0:
                scenario[0, chg1_idx] = delta  # initial month-over-month jump
                scenario[0, chg3_idx] = delta  # 3-month change = the shock
            else:
                scenario[0, chg1_idx] = 0.0  # no further change after the shock
                scenario[0, chg3_idx] = delta if month < 3 else 0.0  # 3-month change fades

            # Predict this month's unemployment
            pred = model.predict(scenario)[0]
            trajectory.append(round(float(pred), 4))

        all_trajectories[name] = trajectory
        results.append({
            "Scenario": name,
            "Rate Change (pp)": delta,
            "New Rate (%)": round(new_rate, 2),
            "Month 1 (%)": trajectory[0],
            "Month 6 (%)": trajectory[5],
            "Month 12 (%)": trajectory[11],
        })
        print(f"  {name:22s}  M1={trajectory[0]:.2f}%  M6={trajectory[5]:.2f}%  M12={trajectory[11]:.2f}%")

    # Save scenario results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "scenario_results.csv", index=False)

    # Save full trajectories for reference
    traj_df = pd.DataFrame({"Month": list(range(1, horizon + 1))})
    for name, traj in all_trajectories.items():
        traj_df[name] = traj  # one column per scenario
    traj_df.to_csv(OUTPUT_DIR / "scenario_trajectories.csv", index=False)

    # Plot: line chart showing diverging trajectories over 12 months
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2ecc71", "#82e0aa", "#f4d03f", "#e67e22", "#e74c3c"]  # green to red
    months = list(range(1, horizon + 1))

    for (name, traj), color in zip(all_trajectories.items(), colors):
        ax.plot(months, traj, color=color, linewidth=2.5, marker="o", markersize=5, label=name)

    # Set the x and y labels and title of the plot
    ax.set_xlabel("Months Forward", fontsize=11)
    ax.set_ylabel("Predicted Unemployment Rate (%)", fontsize=11)
    ax.set_title("Scenario Simulation: 12-Month Unemployment Trajectories\nUnder Different BoC Rate Scenarios", fontsize=13)
    ax.legend(loc="best", fontsize=9)
    ax.set_xticks(months)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scenario_simulation.png", dpi=150)
    plt.close()
    print(f"\n  Saved to {OUTPUT_DIR / 'scenario_simulation.png'}")

    return results_df
