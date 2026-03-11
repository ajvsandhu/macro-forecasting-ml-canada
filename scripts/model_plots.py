# Visualization — plots for model evaluation and interpretation

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # renders plots to file
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from model_data import OUTPUT_DIR

# Shows Ridge coefficients — how each feature pushes unemployment up or down
# Plots through bar chart the coefficients of the Ridge regression model
def plot_ridge_coefficients(pipe, feature_names, filename="ridge_coefficients.png", title=None):
    coefs = pipe.named_steps["ridge"].coef_  # standardized coefficients
    sorted_idx = np.argsort(np.abs(coefs))  # sort by absolute magnitude

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by direction: red = increases unemployment, blue = decreases
    colors = ["#e74c3c" if c > 0 else "#3498db" for c in coefs[sorted_idx]]
    ax.barh(range(len(sorted_idx)), coefs[sorted_idx], color=colors)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel("Standardized Coefficient (effect on unemployment)")
    ax.set_title(title or "Ridge Regression Coefficients\nRed = increases unemployment, Blue = decreases", fontsize=13)
    ax.axvline(x=0, color="black", linewidth=0.5)  # zero line
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved {filename}")


# Evaluation

# Computes RMSE, MAE, and MAPE for one model's predictions
def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # root mean squared error
    mae = mean_absolute_error(y_true, y_pred)  # mean absolute error
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # mean absolute % error
    return {"RMSE": rmse, "MAE": mae, "MAPE (%)": mape}


# Evaluates all models, prints a comparison table, and saves to CSV
def evaluate_all(y_test, predictions):
    print("\n=== Model Evaluation ===\n")

    rows = []
    for name, preds in predictions.items():
        metrics = compute_metrics(y_test, preds)  # calculate 3 error metrics
        metrics["Model"] = name  # tag with model name
        rows.append(metrics)
        print(f"  {name:20s}  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  MAPE={metrics['MAPE (%)']:.2f}%")

    # Combine into a DataFrame and save
    metrics_df = pd.DataFrame(rows)[["Model", "RMSE", "MAE", "MAPE (%)"]]
    metrics_df.to_csv(OUTPUT_DIR / "metrics_comparison.csv", index=False)
    print(f"\n  Saved to {OUTPUT_DIR / 'metrics_comparison.csv'}")

    return metrics_df


# Visualization

# Plots actual vs predicted unemployment: one chart per model (clearer than all-in-one)
def plot_predictions(dates_test, y_test, predictions):
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]
    for (name, preds), color in zip(predictions.items(), colors):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dates_test, y_test, color="black", linewidth=2.5, label="Actual", zorder=5)
        ax.plot(dates_test, preds, color=color, linewidth=2, alpha=0.9, label=name)
        ax.set_title(f"Actual vs {name} — Unemployment Rate (Test Period)", fontsize=13)
        ax.set_xlabel("Date")
        ax.set_ylabel("Unemployment Rate (%)")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        out_path = OUTPUT_DIR / f"predictions_vs_actual_{safe_name}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
    print(f"  Saved {len(predictions)} charts to {OUTPUT_DIR / 'predictions_vs_actual_*.png'}")


# Horizontal bar charts comparing RMSE, MAE, and MAPE across models
def plot_model_comparison(metrics_df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Model Performance Comparison (lower is better)", fontsize=14, fontweight="bold")

    for i, metric in enumerate(["RMSE", "MAE", "MAPE (%)"]):
        sorted_df = metrics_df.sort_values(metric)  # sort so best model is on top
        n = len(sorted_df)
        # Green-to-red gradient: best model gets green, worst gets red
        colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, n))
        axes[i].barh(sorted_df["Model"], sorted_df[metric], color=colors)
        axes[i].set_xlabel(metric)
        axes[i].invert_yaxis()  # put the best (smallest) at the top

        # Label each bar with its value
        for j, val in enumerate(sorted_df[metric]):
            axes[i].text(val, j, f" {val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison_bar.png", dpi=150)
    plt.close()
    print(f"  Saved to {OUTPUT_DIR / 'model_comparison_bar.png'}")


# Shows which features the model relies on most (top 15)
def plot_feature_importance(importances, feature_names, model_name):
    sorted_idx = np.argsort(importances)  # indices that would sort by importance (ascending)
    top_n = min(15, len(feature_names))  # show top 15 (or fewer if less features)
    sorted_idx = sorted_idx[-top_n:]  # take the top N most important

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(top_n), importances[sorted_idx], color="#3498db")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])  # feature names on y-axis
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Features — {model_name}", fontsize=13)
    plt.tight_layout()

    # Save with model name in filename (e.g., feature_importance_random_forest.png)
    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(OUTPUT_DIR / f"feature_importance_{safe_name}.png", dpi=150)
    plt.close()
    print(f"  Saved feature_importance_{safe_name}.png")
