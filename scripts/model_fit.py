# Model fitting, baselines, statistical, and ML models

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
import warnings
warnings.filterwarnings("ignore")

from model_data import TARGET, VAR_COLS

# Baseline models
# Our complex models (ARIMA, XGBoost, etc.) should beat these; if not, they add no value

# Predict next month = last month (e.g., March forecast = February actual)
def baseline_persistence(y_test, X_test, feature_cols):
    lag_idx = feature_cols.index("unemp_lag_1m")  # unemp_lag_1m = last month unemployment (already in our data)
    preds = X_test[:, lag_idx]  # prediction = that value (we assume next month will be the same as last month)
    return preds


# Predict every month = the average unemployment from the training period (same number for all months)
def baseline_mean(y_train, y_test):
    train_mean = y_train.mean()  # average unemployment across all training months (e.g., 7.2%)
    preds = np.full(len(y_test), train_mean)  # fill an array with that number, one slot per test month
    return preds

# ARIMA (AutoRegressive Integrated Moving Average) — forecasts one series using only its own past
# We tune (p, d, q) to fit the model. Each controls a different part of the forecast:
#   (AR) p = how many past unemployment values to use (e.g., p=1 means use last month)
#   (I)  d = how many times to difference (subtracting the previous value from the current value)
#   the  data to remove trends by making the mean and variance stationary/constant (d=0 means no differencing)
#   (MA) q = how many past forecast errors (relationship between an observation and the residual errors)
#   e.g., if we over-predicted last month, the model nudges the next forecast down

# Searches over (p,d,q) orders and returns the one with lowest AIC
def find_best_arima_order(y_train):
    best_aic = np.inf  # start with infinity so any model beats it
    best_order = (1, 0, 0)  # fallback if nothing converges

    # Try all combinations of (p, d, q) and pick the one with lowest AIC
    for p in range(0, 4):  # p: 0–3 past values
        for d in range(0, 2):  # d: 0 or 1 differencing step
            for q in range(0, 3):  # q: 0–2 past forecast errors
                try:
                    model = ARIMA(y_train, order=(p, d, q))  # create ARIMA with this order
                    result = model.fit()  # fit it to training data
                    if result.aic < best_aic:  # keep it if AIC improved
                        best_aic = result.aic
                        best_order = (p, d, q)
                except Exception:
                    continue  # skip orders that fail to converge

    print(f"  Best ARIMA order: {best_order} (AIC={best_aic:.1f})")
    return best_order


# Fits ARIMA and does rolling 1-step-ahead forecasts (refits the model before each new forecast) on the test period
def fit_arima(train_df, test_df):
    print("\n--- ARIMA ---")

    y_train = train_df[TARGET].values  # training unemployment series
    y_test = test_df[TARGET].values  # test unemployment series

    # Search for the best (p,d,q) order on training data only
    best_order = find_best_arima_order(y_train)

    # Combine train + test into one series for rolling access
    full_series = np.concatenate([y_train, y_test])
    train_size = len(y_train)  # index where test period begins
    preds = []  # will hold one prediction per test month

    # Rolling access: Each time we predict a month, we use all past data including months that
    # have already been predicted in pervious iterations

    # Rolling 1-step-ahead: for each test month, fit on all prior data, predict 1 step
    for t in range(len(y_test)):
        history = full_series[:train_size + t]  # all data up to (not including) this test month
        model = ARIMA(history, order=best_order)  # create ARIMA with pre-selected order
        result = model.fit()  # refit on the expanding window
        forecast = result.forecast(steps=1)[0]  # predict 1 month ahead
        preds.append(forecast)

    print(f"  Rolling 1-step-ahead forecast for {len(y_test)} months")
    return np.array(preds)


# VAR (multivariate time series)
# VAR (Vector Autoregression) used for time series forecasting
# Accurately predicts multiple related variables at once by modeling the relationships between them
# Enough variables to capture the relationship between the variables without overfitting

# Fits a Vector Autoregression on differenced macro variables
def fit_var(train_df, test_df):
    print("\n--- VAR ---")

    # Extract the small subset of variables for VAR
    var_train = train_df[VAR_COLS].copy()  # training period only
    var_test = test_df[VAR_COLS].copy()  # test period only

    # Difference all series to ensure stationarity (subtract previous month's value)
    var_train_diff = var_train.diff().dropna()  # loses first row to differencing

    # Select the best lag order by AIC (try up to 12 monthly lags)
    model = VAR(var_train_diff)
    lag_order_result = model.select_order(maxlags=12)
    best_lag = lag_order_result.aic  # number of lags that minimizes AIC
    best_lag = max(best_lag, 1)  # use at least 1 lag (0 would mean no model)
    print(f"  Best VAR lag order: {best_lag} (by AIC)")

    # Fit the VAR model with the selected lag order
    result = model.fit(best_lag)

    # Rolling access: Each time we predict a month, we use all past data including months that
    # have already been predicted in pervious iterations

    # Combine train + test for rolling access to actual values
    var_full = pd.concat([var_train, var_test], axis=0)
    var_full_diff = var_full.diff().dropna()  # differenced version of full series

    # Where the test period starts in the differenced series
    test_start_diff = len(var_train_diff)

    # Rolling 1-step-ahead forecast on the differenced series
    preds_diff = []
    for t in range(len(test_df)):
        history_end = test_start_diff + t  # current position in differenced series
        # Take the last 'best_lag' differenced observations as input
        lagged = var_full_diff.iloc[history_end - best_lag:history_end].values
        fc = result.forecast(lagged, steps=1)  # forecast 1 step ahead (differenced)
        preds_diff.append(fc[0])

    preds_diff = np.array(preds_diff)  # shape: (n_test, n_var_cols)

    # Converts the differences (after a month, 0.4% increase) back to levels (e.g. in the next month 7.6% unemployment rate)
    # Levels are the actual unemployment rates, not the differences allowing us to predict the actual unemployment rate
    unemp_idx = VAR_COLS.index("unemployment_rate")  # column index within VAR_COLS
    preds_level = []
    for t in range(len(test_df)):
        if t == 0:
            prev_level = train_df[TARGET].iloc[-1]  # last training value
        else:
            prev_level = test_df[TARGET].iloc[t - 1]  # previous test month's actual
        pred = prev_level + preds_diff[t, unemp_idx]  # level = previous + predicted change
        preds_level.append(pred)

    print(f"  Rolling 1-step-ahead forecast for {len(test_df)} months")
    return np.array(preds_level)


# Random Forest
# Makes multiple model that each predict using random subset of data and features
# Good for complex data with many features and relationships with it being robust overfitting

# Tunes and fits a Random Forest regressor with time-series cross-validation
def fit_random_forest(X_train, y_train, X_test):
    print("\n--- Random Forest ---")

    # Hyperparameter grid to search over (every number explained)
    param_grid = {
        "n_estimators": [200, 500],  # 200 or 500 trees
        "max_depth": [5, 10, None],  # depth of each tree (number of edges from root to leaf), 5, 10 , None = unlimited (can overfit)
        "min_samples_leaf": [2, 5],  # minimum number of samples required at each leaf, 2 or 5 — min samples at each leaf; higher = simpler tree
    }

    # Time-series cross-validation: 5 folds that respect temporal order (train before test)
    tscv = TimeSeriesSplit(n_splits=5)

    # Grid search: tries every param combination, picks the one with lowest MSE
    rf = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(
        rf, param_grid,
        cv=tscv,  # time-series aware cross-validation
        scoring="neg_mean_squared_error",  # sklearn convention: higher = better, so MSE is negated
        n_jobs=-1  # use all CPU cores for parallel fitting
    )
    grid.fit(X_train, y_train)  # run the search

    best_rf = grid.best_estimator_  # the model with the best CV score
    print(f"  Best params: {grid.best_params_}")
    print(f"  CV RMSE: {np.sqrt(-grid.best_score_):.4f}")  # convert neg MSE back to RMSE

    preds = best_rf.predict(X_test)  # predict on the held-out test set
    return preds, best_rf


# XGBoost
# Builds trees one at a time, each new tree corrects the errors of the previous ones (gradient boosting)
# Good for complex patterns and often wins competitions; uses shallow trees to avoid overfitting

# Tunes and fits XGBoost regressor with time-series cross-validation
def fit_xgboost(X_train, y_train, X_test):
    print("\n--- XGBoost ---")

    # Hyperparameter grid to search over (every number explained)
    param_grid = {
        "n_estimators": [200, 500],  # 200 or 500 boosting rounds — each round adds one tree
        "max_depth": [3, 5],  # 3 or 5 — max levels per tree; XGBoost usually works better with shallow trees
        "learning_rate": [0.05, 0.1],  # 0.05 = 5% or 0.1 = 10% — how much each tree contributes; smaller = slower but more careful
    }

    # Time-series cross-validation: 5 folds that respect temporal order
    tscv = TimeSeriesSplit(n_splits=5)

    # Grid search over all parameter combinations
    xgb = XGBRegressor(random_state=42, verbosity=0)  # verbosity=0 suppresses training logs
    grid = GridSearchCV(
        xgb, param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_xgb = grid.best_estimator_
    print(f"  Best params: {grid.best_params_}")
    print(f"  CV RMSE: {np.sqrt(-grid.best_score_):.4f}")

    preds = best_xgb.predict(X_test)
    return preds, best_xgb


# Ridge Regression
# Fits a linear model (weighted sum of features) with L2 regularization (adds a penalty (sum of squared coefficients)
# to the model to prevent overfitting) to shrink coefficients and prevent overfitting
# Good for interpretable coefficients and extrapolating to new scenarios (e.g. "what if rates change?")

# Fits Ridge regression with standardized features
def fit_ridge(X_train, y_train, X_test):
    print("\n--- Ridge Regression ---")

    # Pipeline: scale features first (Ridge is sensitive to different feature scales)
    pipe = Pipeline([
        ("scaler", StandardScaler()),  # z-score normalization: (x - mean) / std
        ("ridge", Ridge(alpha=1.0)),  # mild L2 regularization prevents overfitting
    ])
    pipe.fit(X_train, y_train) # fit scaler and Ridge together

    preds = pipe.predict(X_test)  # predict on test set (scaler applied automatically)

    # Extract standardized coefficients (show relative feature importance)
    coefs = pipe.named_steps["ridge"].coef_
    print(f"  Fitted with {len(coefs)} features")

    return preds, pipe
