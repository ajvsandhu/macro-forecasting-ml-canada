# Modeling — baseline, statistical, and ML models for unemployment forecasting

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from model_data import OUTPUT_DIR, load_and_split
from model_fit import (
    baseline_persistence,
    baseline_mean,
    fit_arima,
    fit_var,
    fit_random_forest,
    fit_xgboost,
    fit_ridge,
)
from model_plots import (
    evaluate_all,
    plot_predictions,
    plot_model_comparison,
    plot_feature_importance,
    plot_ridge_coefficients,
)
from model_scenarios import simulate_scenarios

# Main

# Orchestrates all modeling steps: split, train, evaluate, visualize, simulate
def main():
    # Create the output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data and split into train/test (chronological 80/20)
    (train_df, test_df, X_train, X_test, y_train, y_test,
     dates_train, dates_test, feature_cols) = load_and_split()

    # Dictionary to store each model's test predictions (order matters for plotting)
    predictions = {}

    # Baselines (set the floor for what a "real" model needs to beat)
    print("\n--- Baselines ---")
    predictions["Persistence"] = baseline_persistence(y_test, X_test, feature_cols)
    predictions["Historical Mean"] = baseline_mean(y_train, y_test)

    # Loading the Statistical models
    predictions["ARIMA"] = fit_arima(train_df, test_df)
    predictions["VAR"] = fit_var(train_df, test_df)

    # Loading the ML models
    rf_preds, rf_model = fit_random_forest(X_train, y_train, X_test)
    predictions["Random Forest"] = rf_preds

    xgb_preds, xgb_model = fit_xgboost(X_train, y_train, X_test)
    predictions["XGBoost"] = xgb_preds

    ridge_preds, ridge_pipe = fit_ridge(X_train, y_train, X_test)
    predictions["Ridge"] = ridge_preds

    # Evaluating all models, computing the metrics (RMSE, MAE, MAPE)
    metrics_df = evaluate_all(y_test, predictions)

    # Generating all plots, plotting the predictions, the model comparison, the feature importance, and the ridge coefficients
    print("\n=== Generating Plots ===")
    plot_predictions(dates_test, y_test, predictions)
    plot_model_comparison(metrics_df)
    plot_feature_importance(rf_model.feature_importances_, feature_cols, "Random Forest")
    plot_feature_importance(xgb_model.feature_importances_, feature_cols, "XGBoost")
    plot_ridge_coefficients(ridge_pipe, feature_cols)

    # Saving all predictions to one CSV for easy inspection
    pred_df = pd.DataFrame({"date": dates_test, "actual": y_test})
    for name, preds in predictions.items():
        pred_df[name] = preds  # add each model's predictions as a column
    pred_df.to_csv(OUTPUT_DIR / "all_predictions.csv", index=False)
    print(f"  Saved all predictions to {OUTPUT_DIR / 'all_predictions.csv'}")

    # Scenario simulation: "what if" BoC cuts/hikes rates? Forecast models lean the last month's unemployment rate
    # to calculate the unemployment rate for the next month. This prediction is made by the last month not by just the interest rate.
    # To find this out, we need to train a model that predicts the unemployment rate when the unemployment rate is fixed.
    print("\n--- Training scenario model (Ridge without unemp_lag_1m) ---")
    scenario_cols = [c for c in feature_cols if c != "unemp_lag_1m"]  # drop the AR feature
    X_train_sc = train_df[scenario_cols].values  # training features for scenario model
    X_test_sc = test_df[scenario_cols].values  # test features for scenario model

    # Fit Ridge for scenario analysis to predict the unemployment rate when the unemployment rate is fixed.
    ridge_scenario = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    ridge_scenario.fit(X_train_sc, y_train)
    sc_rmse = np.sqrt(mean_squared_error(y_test, ridge_scenario.predict(X_test_sc)))
    print(f"  Scenario model RMSE: {sc_rmse:.4f} (higher than forecasting models — expected)")
    plot_ridge_coefficients(
        ridge_scenario, scenario_cols,
        filename="scenario_coefficients.png",
        title="Scenario Model Coefficients (without unemp_lag_1m)\nRed = increases unemployment, Blue = decreases"
    )

    simulate_scenarios(ridge_scenario, X_test_sc, scenario_cols)

    print("\n=== Step 5 Complete ===")
    print(f"All outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
