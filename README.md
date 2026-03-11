# macro-forecasting-ml-canada

Forecasting Canadian unemployment using macroeconomic indicators. Full methodology, results, and conclusions are in [report.md](report.md).

## Setup and run

Outputs (plots, CSVs) are not committed. To produce them:

```bash
pip install -r requirements.txt
python scripts/pull_data.py
python scripts/process_data.py
python scripts/eda.py
python scripts/model.py
```

- `outputs/eda/` — summary stats, correlation matrix, distributions, stationarity tests, lagged correlation, overlay plots
- `outputs/modeling/` — metrics, predictions vs actual (per model), feature importance, Ridge coefficients, scenario simulation

## Project structure

- `scripts/` — pull_data.py, process_data.py, eda.py, model.py (and model_*.py modules)
- `data/raw/` — raw data (StatCan, BoC, FRED)
- `data/processed/` — merged_monthly.csv
- `report.md` — full research report
