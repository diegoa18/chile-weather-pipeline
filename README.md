# Chile Weather Pipeline

Weather forecasting pipeline for Chilean cities using Open-Meteo data + XGBoost.

## Usage

```bash
python3 main.py train     # Download → clean → train → validate → save model + metrics
python3 main.py predict   # Load model → forecast 3 days → save CSV + plot
```

## Output

```
data/<city>/
├── plots/          temperature_trend.png, precipitation.png (train)
│                   forecast.png (predict)
├── results/        metrics.json, eda.json, model_metrics.json, forecast.csv
├── models/         <city>_temp_model.pkl
├── raw/            raw API data
└── processed/      cleaned data
```

## Configuration

Edit `src/config/settings.py`:

- `CITIES` — add or remove cities
- `DEFAULT_DAYS_BACK` — training window in days (default: 365)
- `DEFAULT_FORECAST_DAYS` — prediction horizon (default: 3)
- `MODEL_PARAMS_XGB` / `MODEL_PARAMS_RF` — hyperparameters

## Tests

```bash
python3 -m pytest tests/
```
