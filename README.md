# Predictive Sales Forecasting & Business Analytics System

An end-to-end project that cleans retail sales data, trains and compares forecasting models,
produces a multi-month revenue forecast, and exposes everything through a **Streamlit dashboard**
and a **FastAPI REST service**.

## Features

- **Data layer** – validates and cleans a transactional sales CSV; ships with a realistic synthetic
  dataset generator (5 stores × 5 categories, ~4.7 years of daily data, trend + weekly/yearly
  seasonality + festive uplift + promotions).
- **Feature engineering** – calendar features, lags (1, 7, 14, 28 and 364 days), rolling means, promotion share.
- **Modelling** – Seasonal-Naive baseline vs Ridge, Random Forest and Gradient Boosting.
  Models are compared with a **recursive rolling-origin backtest** (realistic multi-step error),
  and the best one is refit on all data and saved.
- **Forecasting** – recursive day-by-day forecast with an approximate 95% interval and a
  "planned promotion share" what-if input.
- **Business analytics** – KPIs (revenue, MoM/YoY growth), monthly trend, store & category
  performance, ABC analysis, weekday pattern, store × category heatmap, promotion lift.
- **Dashboard** (Streamlit) and **API** (FastAPI, auto docs at `/docs`).
- **Tests** (pytest) covering data, features, pipeline and API.

## Project structure

```
sales-forecasting-system/
├── README.md
├── requirements.txt
├── .gitignore
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI service
├── app/
│   └── dashboard.py         # Streamlit dashboard
├── data/
│   ├── raw/                 # sales.csv (generated or your own)
│   └── processed/           # daily_sales.csv
├── models/                  # best_model.joblib
├── notebooks/               # your own EDA notebooks
├── reports/                 # metrics.json, forecast.csv, PNG charts
├── src/
│   ├── config.py            # paths & modelling settings
│   ├── data_generator.py    # synthetic data
│   ├── data_loader.py       # load, validate, clean, aggregate
│   ├── features.py          # feature engineering
│   ├── models.py            # training, backtesting, selection
│   ├── forecast.py          # recursive multi-step forecasting
│   ├── analytics.py         # KPIs and business analytics
│   ├── visualize.py         # static PNG charts
│   └── pipeline.py          # end-to-end CLI
└── tests/
    └── test_system.py
```

## Setup

Requires **Python 3.10+** (developed on 3.13).

**Windows (PowerShell)**
```powershell
cd sales-forecasting-system
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry.

**macOS / Linux**
```bash
cd sales-forecasting-system
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the application

Run everything from the project root (the folder containing `README.md`), with the virtual environment active.

**1. Build data, train models, create the forecast** (~15 seconds)
```bash
python -m src.pipeline --regenerate
```
This writes `data/raw/sales.csv`, `data/processed/daily_sales.csv`, `models/best_model.joblib`,
`reports/metrics.json`, `reports/forecast.csv` and three PNG charts.
Options: `--horizon 120` (forecast days), `--no-plots`. Omit `--regenerate` to reuse the existing CSV.

**2. Launch the dashboard**
```bash
streamlit run app/dashboard.py
```
Opens at http://localhost:8501. Use the sidebar to filter by date, store and category; the
**Forecast** tab has sliders for horizon and planned promotion share.

**3. Launch the API** (second terminal, same venv)
```bash
uvicorn api.main:app --reload
```
Interactive docs: http://127.0.0.1:8000/docs

| Endpoint | Description |
|---|---|
| `GET /health` | service and artifact status |
| `GET /kpis` | headline KPIs |
| `GET /metrics` | model comparison and feature importance |
| `GET /history?days=90` | recent daily revenue |
| `GET /forecast?days=30&promo_share=0.12` | revenue forecast with intervals |
| `GET /analytics/{monthly\|category\|store\|weekday\|promo}` | analytics tables |

Example: `curl "http://127.0.0.1:8000/forecast?days=14"`

**4. Run the tests**
```bash
pytest -q
```

## Using your own data

Replace `data/raw/sales.csv` with a CSV (one row per date × store × category) and re-run
`python -m src.pipeline` (without `--regenerate`).

| Column | Required | Notes |
|---|---|---|
| `date` | yes | e.g. `2025-03-14` |
| `store` | yes | any label |
| `category` | yes | any label |
| `units` | yes | units sold |
| `revenue` | yes | sales amount |
| `unit_price` | no | derived from revenue/units if missing |
| `promo` | no | 0/1 flag, defaults to 0 |

You need roughly **2 years or more** of daily history (the model uses a 364-day lag and a 120-day hold-out).

## How the forecast works

1. Transactions are aggregated to total daily revenue; missing days are interpolated.
2. Features for day *t* use only information available before *t* (lags, rolling means) plus
   calendar fields and the planned promotion share.
3. Each model is trained on everything except the last 120 days, then scored two ways:
   *one-step-ahead* (uses true yesterday values, optimistic) and *recursive* (three
   30-day forecasts that feed their own predictions back in, which is how real forecasting works).
   The best model by recursive RMSE is refit on all data.
4. Intervals are forecast ± 1.96 × the backtest RMSE (an approximation, constant over the horizon).

Reference results on the bundled synthetic data: the selected model reaches about 4.8% recursive
MAPE on the hold-out versus 6.7% for the seasonal-naive baseline.

## Known limitations

- The sample data is synthetic; accuracy on real data will differ. Re-check the metrics on yours.
- Forecasts are for **total** revenue (all stores/categories combined), not per store or SKU.
- Peak-season demand is slightly under-forecast (about 6% low in a stress test on Oct–Nov of previous years).
- Holidays are captured only through month/day-of-year features and the year-ago lag; add a
  holiday calendar for better festival-date accuracy.
- Promotion share for future dates is an assumption you provide (default = last year's average).

## Push the project to GitHub

1. **Install Git** (https://git-scm.com/downloads) and create a GitHub account.
2. **Create an empty repository** at https://github.com/new — name it `sales-forecasting-system`,
   and do **not** tick "Add a README", ".gitignore" or "license" (the project already has files).
3. **Initialise and commit locally** (from the project root):
   ```bash
   git init -b main
   git add .
   git status            # check: no .venv, no .csv/.joblib files should be listed
   git commit -m "Initial commit: predictive sales forecasting system"
   ```
   If Git says it doesn't know who you are:
   ```bash
   git config --global user.name  "Your Name"
   git config --global user.email "you@example.com"
   ```
4. **Connect and push** (replace `<your-username>`):
   ```bash
   git remote add origin https://github.com/<your-username>/sales-forecasting-system.git
   git push -u origin main
   ```
   When prompted to sign in: GitHub no longer accepts account passwords on the command line.
   Use the browser sign-in from Git Credential Manager (included with Git for Windows), or a
   **Personal Access Token** (GitHub → Settings → Developer settings → Personal access tokens,
   scope `repo`) as the password. SSH keys also work.
5. **Shortcut with GitHub CLI** (does steps 2 and 4 in one go):
   ```bash
   gh auth login
   gh repo create sales-forecasting-system --public --source=. --remote=origin --push
   ```
6. **Later changes:**
   ```bash
   git add .
   git commit -m "Describe your change"
   git push
   ```

Common errors

| Message | Fix |
|---|---|
| `remote origin already exists` | `git remote set-url origin <url>` |
| `src refspec main does not match any` | you have no commit yet, or your branch is `master`: run `git commit ...` then `git branch -M main` |
| `failed to push some refs` (remote has files) | `git pull origin main --rebase`, then `git push` |
| `Authentication failed` | use a Personal Access Token instead of your password |

Generated data, models and reports are git-ignored on purpose; anyone who clones the repo
recreates them with `python -m src.pipeline --regenerate`.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`** – run commands from the project root, not from inside `src/` or `app/`.
- **Dashboard shows "Sales data not found"** – run the pipeline first (step 1).
- **API returns 503** – the model or data is missing; run the pipeline.
- **Port already in use** – `streamlit run app/dashboard.py --server.port 8502` or `uvicorn api.main:app --port 8001`.
