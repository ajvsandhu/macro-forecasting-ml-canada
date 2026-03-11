# Forecasting Canadian Unemployment Using Macroeconomic Indicators

To reproduce all outputs:

```bash
pip install -r requirements.txt
python scripts/pull_data.py
python scripts/process_data.py
python scripts/eda.py
python scripts/model.py
```

Outputs are written to `outputs/eda/` and `outputs/modeling/`.

---

## 1. Executive Summary

I was exploring the idea whether Canadian unemployment can be reliably forecasted using macroeconomic indicators, and how monetary policy influences short term labor market dynamics. Using monthly data from 2009(earliest it was avaliable) to 2024, I built seven forecasting models—baselines (Persistence, Historical Mean), statistical (ARIMA, VAR), and machine learning (Random Forest, XGBoost, Ridge). **Main result:** Persistence achieves the lowest RMSE (0.23) and MAPE (2.7%), followed closely by Ridge (0.24, 3.6%). The complex ML models do not outperform these simpler approaches. **Scenario simulation** with a Ridge model trained without the unemployment lag shows that rate cuts raise predicted unemployment over 12 months, consistent with a delayed monetary policy transmission. **Takeaway:** Next month's unemployment usually looks like this month's. Adding macro variables improves it very minimally. The indicators that predict unemployment best are employment rate, interest rate changes, GDP, and inflation. The scenario model shows that interest rates and unemployment move together in a sensible way, which helps interpret policy.


## 2. Introduction

### Research Question

Can Canadian unemployment be reliably forecasted using macroeconomic indicators, and how does monetary policy influence short-term labor market dynamics?

### Motivation

Unemployment is a key indicator of economic health and a primary target of monetary policy. Forecasting it helps policymakers, businesses, and researchers anticipate labor market conditions. Understanding how the Bank of Canada’s overnight rate affects unemployment—with lags and through various channels—is essential for interpreting policy decisions and their impact.

### Scope

- **Data range:** 2009-04 to 2024-12 (177 months after preprocessing)
- **Models:** Persistence, Historical Mean, ARIMA(1,0,2), VAR(2), Random Forest, XGBoost, Ridge
- **Scenario design:** Five BoC rate scenarios (−200bp, −100bp, status quo, +100bp, +200bp) over a 12 month horizon, using a Ridge model without the unemployment lag to isolate macro effects



## 3. Data

### Target Variable

**Unemployment rate:** The share of the labour force (people 15+ who are employed or actively seeking work) that is without a job. Data are monthly and seasonally adjusted. It is the main indicator of labour market health and a key target for monetary policy. Forecasting it helps policymakers, firms, and households anticipate economic conditions.

### Features: What Each Variable Means and Why It Matters

**Levels (raw values):**

- **Overnight rate:** The Bank of Canada’s policy rate. It affects borrowing costs and economic activity; higher rates can slow growth and raise unemployment.
- **Employment rate:** Share of the working age population that is employed. It moves inversely with unemployment and is a direct measure of labour market strength.
- **CPI:** Consumer Price Index, a measure of inflation. High inflation can lead to rate hikes, which may slow growth and raise unemployment.
- **GDP:** Total output of the economy. Growth tends to support job creation; recessions raise unemployment.
- **Bond yield:** Return on government bonds. Reflects expectations about rates, inflation, and growth.
- **Exchange rate (CAD/USD):** Value of the Canadian dollar. Affects exports, imports, and employment in trade exposed sectors.
- **Oil price:** Price of crude oil. Important for Canada as an oil producer; swings affect energy sector investment and jobs.

**Lags (past values):**

- **rate_lag_3m, rate_lag_6m, rate_lag_12m:** Overnight rate 3, 6, and 12 months ago. Monetary policy affects the economy with a lag; these capture delayed effects.
- **unemp_lag_1m:** Unemployment from the previous month. Unemployment is highly persistent, so last month’s value is a strong predictor.

**Changes (rates of change):**

- **rate_change_1m, rate_change_3m:** Change in the overnight rate over 1 and 3 months. Capture recent shifts in monetary policy.
- **cpi_change_1m, cpi_change_12m:** Monthly and annual inflation. Rising inflation may prompt rate hikes and affect the labour market.
- **gdp_change_1m, gdp_change_12m:** Monthly and annual GDP growth. Direct measures of economic momentum and job creation.
- **oil_change_3m, fx_change_3m:** 3 month change in oil price and exchange rate. Reflect recent shocks to energy and trade.

**Rolling (moving averages):**

- **rate_rolling_6m:** 6 month average of the overnight rate. Smooths short term noise and highlights sustained policy stance.
- **unemp_rolling_12m:** 12 month average of unemployment. Highlights longer run labour market trends.

**Derived:**

- **yield_spread (10y − 3m):** Difference between 10 year and 3 month bond yields. The gap between long term and short term interest rates and when it turns negative, it often signals a higher chance of recession and rising unemployment.

### Sample

- **Merged range:** 2009-04 to 2024-12
- **Final sample:** 177 rows, 25 columns (after dropping rows with NaN from lags)

The data comes from three main sources. **Statistics Canada** provides unemployment, employment, CPI, and GDP. **Bank of Canada** provides the overnight rate and 10-year bond yield. **FRED** (Federal Reserve Economic Data) provides the CAD/USD exchange rate and WTI oil price. Each row is one month; all series are merged by date so that every row has values from all sources. Rows with missing values (from lagged features that need prior months) are dropped, leaving 177 complete months.


## 4. Exploratory Analysis

### Main Correlations with Unemployment

Positive means it most with unemployement while neagtive means it moves opposite  
The strongest predictors of unemployment are:
- last month's unemployment (0.93)
- employment rate (−0.85)
- 12 month rolling unemployment (0.68)
- 3 month rate change (−0.58)
- GDP (−0.56)
- 12 month CPI change (−0.55)
- and overnight rate (−0.50)

Employment rate is the strongest macro predictor (best among external economic indicators) (r = −0.85). The overnight rate is negatively correlated at lag 0 (r ≈ −0.50).

### Lagged Effects

Overnight rate correlations with unemployment shift from negative at short lags to positive at longer lags, consistent with the idea that rate hikes initially cool the economy (lower unemployment) but later lead to higher unemployment as the economy adjusts. This is because if the unemployement is already low the BOC will raise rates to foucs more on inflation then unemployement. Because mometary policies take time to take affect the unemployement wont be affected until months later. If you look at long lags the correlation actaully becomes positive supporting my theory.

### Stationarity and Multicollinearity

**Stationarity (ADF test):** Before modeling, we need to know whether a time series is stationary—meaning its mean and variance stay roughly constant over time, with no long run trend. Non stationary series (e.g. GDP or CPI that trend upward) can produce misleading correlations and unreliable forecasts.

The Augmented Dickey-Fuller (ADF) test checks for a "unit root": a statistical property that makes a series non stationary. The null hypothesis is that the series has a unit root (i.e. is non stationary). If the p-value is below 0.05, we reject the null and conclude the series is stationary. If p ≥ 0.05, we fail to reject, and the series is treated as non stationary.

**Results:** Unemployment and employment rates are stationary (p < 0.05), so we can use them in levels. CPI, GDP, overnight rate, bond yield, exchange rate, and oil price are non stationary; for these we use differenced or change versions (e.g. month over month or year over year changes) where appropriate, which removes the trend and makes the series suitable for modeling.

**Multicollinearity:** CPI and GDP are highly correlated (r ≈ 0.96) , as seen in the correlation matrix. If we don’t handle multicollinearity, the model’s coefficients can be unstable and hard to interpret, and forecasts can be less reliable. We therefore use Ridge or tree based methods, which handle correlated features better than plain linear regression.

### Figures (from outputs/eda/)

- **Figure 1:** `correlation_matrix.png` — Full correlation matrix of variables
- **Figure 2:** `lagged_correlation.png` — Correlation of overnight rate with unemployment at different lags
- **Figure 3:** `distributions.png` — Distributions of key variables


## 5. Methods

### Train/Test Split

- **Chronological 80/20 split** (no shuffling; time series order preserved)
- Train: 141 months (2010-04 to 2021-12)
- Test: 36 months (2022-01 to 2024-12)
- 21 features (after removing date, target, and leaky columns)

### Data Leakage Handling

Two features were excluded because they contain current month unemployment information:

- `unemp_change_1m` — current − previous month (leaks current value)
- `unemp_rolling_12m` — 12 month rolling mean including current month

### Models

**Persistence (Baseline)** — The baseline assumes next month = this month's unemployment. Used as a simple benchmark: if complex models can't beat "no change," they add no value. Unemployment typically changes each month, but the changes are usually small. Persistence is often wrong (it predicts no change when there is a change). Because unemployment is highly persistent (correlation 0.93 with last month), this is a strong baseline beacuse the smaller the change the better it is.

**Historical Mean (Baseline)** — Forecasts every month = the average unemployment over the training period (a flat line). No tuning. It performs poorly when unemployment has a trend: if unemployment is rising or falling, the actual values move away from that average, so the baseline fails. This matters because if Historical Mean did well, unemployment would just bounce around randomly—nothing to predict. Because it does poorly, we know unemployment moves in patterns (up, down, or persistent). Those patterns mean better models have something to learn.

**ARIMA (Statistical)** — Autoregressive Integrated Moving Average. A univariate model that uses only past unemployment values to predict the next month—no macro variables. It combines an autoregressive part (past values), optional differencing (to remove trends), and a moving average part (past forecast errors). The order (p,d,q) is chosen by searching combinations and picking the one with lowest AIC, we selected (1,0,2). Rolling 1 step ahead: for each test month, refit on all prior data and predict one step. Used to see how well unemployment forecasts using only its own history, without any macro indicators—a pure time series benchmark.

**VAR (Statistical)** — Vector Autoregression. Models multiple series together (unemployment, employment, overnight rate, bond yield) by differencing them for stationarity and using lagged values of all variables. Each variable is predicted from its own lags and the lags of the others, capturing how they influence each other. Lag order (2) was chosen by AIC. Rolling 1 step ahead forecasts on the differenced series, then converted back to levels. Used to capture joint dynamics between unemployment and key macro variables—how they move together over time when we allow feedback between them.

**Random Forest (ML)** — Ensemble of decision trees, each tree predicts using a random subset of data and features, and the final prediction is the average across trees. This reduces overfitting because no single tree sees all the data. Tuned via GridSearchCV with TimeSeriesSplit (5 folds): n_estimators [200, 500], max_depth [5, 10, None], min_samples_leaf [2, 5]. Used to capture non linear relationships and feature interactions that linear models miss; the ensemble smooths out individual tree errors.

**XGBoost (ML)** — Gradient boosting: builds trees one at a time, each correcting the errors of the previous ones. Unlike Random Forest (parallel trees), trees are built sequentially so each new tree targets the residuals of the ensemble so far. Tuned via GridSearchCV with TimeSeriesSplit: n_estimators [200, 500], max_depth [3, 5], learning_rate [0.05, 0.1]. Shallow trees (max_depth 3–5) help avoid overfitting. Used for potentially stronger non linear forecasting; often competitive on tabular data.

**Ridge (ML)** — Linear model (weighted sum of features) with L2 regularization that shrinks coefficients toward zero, which helps with multicollinearity. Features are standardized before fitting so the penalty is applied fairly. Alpha = 1.0 (fixed). Used for interpretable coefficients (we can see which features push unemployment up or down) and because it extrapolates well to new scenarios (e.g. "what if rates change?"), unlike tree models that cannot extend beyond the training range.

### Scenario Model

A separate Ridge model is trained **without** `unemp_lag_1m` for the scenario simulation. For scenarios we ask "what if the BoC raises or cuts rates?", in that hypothetical world we don't know the true unemployment path, so we drop unemp_lag_1m to force the model to learn macro→unemployment links from rates, employment, GDP, etc. We use Ridge because it is linear and can extrapolate to rate levels outside the training range; tree models partition the feature space and predict within regions, so they don't extrapolate well. We simulate five BoC rate scenarios (−200bp, −100bp, status quo, +100bp, +200bp) over 12 months, updating rate-related features each month to see how unemployment would respond under different policy paths.

### Evaluation Metrics for models (comparing it to the actaul data)

- **RMSE** — Root mean squared error (penalizes large errors; lower = better predictions)
- **MAE** — Mean absolute error (average error size, easy to interpret, e.g. "off by 0.2%")
- **MAPE** — Mean absolute percentage error (%) (error as % of actual; comparable across scales)

## 6. Results

### Forecast Performance

|      Model      | RMSE  |  MAE  | MAPE (%) |
|-----------------|-------|-------|----------|
|   Persistence   | 0.233 | 0.156 |   2.71   |
|      Ridge      | 0.245 | 0.201 |   3.57   |
|      ARIMA      | 0.338 | 0.236 |   4.26   |
|       VAR       | 0.455 | 0.364 |   6.56   |
|  Random Forest  | 0.476 | 0.391 |   7.32   |
|     XGBoost     | 0.479 | 0.421 |   7.66   |
| Historical Mean | 1.645 | 1.531 |   28.38  |

### Best vs Worst Models

- **Best:** Persistence (RMSE 0.23) and Ridge (0.24) outperform all others. Persistence exploits the high autocorrelation of unemployment; Ridge adds modest improvement with interpretable coefficients.
- **Worst:** Historical Mean (RMSE 1.64) is far worse than all others, confirming that unemployment is not well approximated by a constant.
- **Complex models underperform:** ARIMA, VAR, Random Forest, and XGBoost all do worse than Persistence and Ridge at this 1-month horizon, suggesting limited gain from additional complexity.

### Predictions vs Actual

`outputs/modeling/predictions_vs_actual_*.png`: per model plots. Persistence and Ridge track actual unemployment closely; VAR and tree models show more deviation.

### Feature Importance

- **Ridge coefficients:** Employment rate, overnight rate, and rate changes have strong effects. See `ridge_coefficients.png` and `scenario_coefficients.png`.
- **Tree models:** `unemp_lag_1m` dominates; employment rate and rate related features also matter. See `feature_importance_random_forest.png` and `feature_importance_xgboost.png`.

### Scenario Simulation

12-month unemployment trajectories under different BoC rate scenarios (Ridge without `unemp_lag_1m`):

| Scenario | Rate Change | New Rate (%) | Month 1 (%) | Month 6 (%) | Month 12 (%) |
|----------|-------------|--------------|-------------|-------------|--------------|
| Rate cut -200bp | −2.0 pp | 1.43 | 7.40 | 7.45 | 7.44 |
| Rate cut -100bp | −1.0 pp | 2.43 | 7.00 | 7.03 | 7.02 |
| Status quo | 0.0 pp | 3.43 | 6.60 | 6.60 | 6.60 |
| Rate hike +100bp | +1.0 pp | 4.43 | 6.20 | 6.17 | 6.18 |
| Rate hike +200bp | +2.0 pp | 5.43 | 5.79 | 5.74 | 5.75 |

**Interpretation:** Rate cuts are associated with higher predicted unemployment (e.g., −200bp → 7.4% vs 6.6% status quo), consistent with the idea that cuts often occur when the economy is weak and unemployment is already high or rising. Rate hikes are associated with lower predicted unemployment, consistent with tightening when the economy is strong. The model captures macro→unemployment links in a directionally plausible way.



## 7. Discussion

### Can We Reliably Forecast?

Yes, in a limited sense. Persistence and Ridge achieve RMSE ~0.23–0.24 and MAPE ~2.7–3.6%, indicating that 1 month ahead unemployment can be forecast with reasonable accuracy. However, the best performance comes from simple models (Persistence, Ridge); complex ML and multivariate models do not improve forecasts at this horizon. Unemployment is highly persistent, so "next month ≈ this month" is hard to beat.

### How Does Monetary Policy Matter?

Scenario simulation suggests that the overnight rate is associated with unemployment in a directionally consistent way: cuts with higher unemployment, hikes with lower. This reflects both the timing of policy (cuts when unemployment is high) and the model’s learned macro→unemployment links. The Ridge model without the unemployment lag allows extrapolation to hypothetical rate paths, providing a tool for "what if" policy analysis.

### Limitations

- **Sample size:** 177 months limits the power of complex models.
- **Stationarity:** Several macro series are non-stationary; differencing and changes help but may not fully address structural breaks.
- **Extrapolation:** Scenario results rely on Ridge extrapolating to rate levels outside the training range; actual responses may differ.
- **Horizon:** Results apply to 1-month-ahead forecasts; longer horizons may behave differently.

### Possible Extensions

- Add more predictors (e.g., sectoral employment, vacancies).
- Extend to multi-step (3-, 6-, 12-month) forecasts.
- Compare with professional forecasts (e.g., BoC, private sector).
- Use alternative scenario designs (e.g., gradual rate paths).



## 8. Conclusion

Canadian unemployment can be forecasted with reasonable accuracy at a 1 month horizon. Persistence and Ridge achieve the best performance (RMSE ~0.23–0.24); more complex models do not improve on these. Scenario simulation with a Ridge model trained without the unemployment lag indicates that monetary policy is associated with unemployment in a directionally plausible way: rate cuts with higher unemployment, rate hikes with lower. The project demonstrates that simple, interpretable models can be effective for short horizon unemployment forecasting and for exploring the macro→unemployment relationship under hypothetical policy scenarios.


### Full Metrics Table

|      Model      |  RMSE  |  MAE   | MAPE (%) |
|-----------------|--------|--------|----------|
|   Persistence   | 0.2333 | 0.1556 |   2.71   |
| Historical Mean | 1.6448 | 1.5313 |   28.4   |
|      ARIMA      | 0.3382 | 0.2364 |   4.26   |
|       VAR       | 0.4553 | 0.3645 |   6.56   |
|  Random Forest  | 0.4765 | 0.3906 |   7.32   |
|     XGBoost     | 0.4793 | 0.4214 |   7.66   |
|      Ridge      | 0.2447 | 0.2005 |   3.57   |
