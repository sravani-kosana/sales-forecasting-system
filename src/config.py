"""Central configuration.

Other modules access these values as ``config.NAME`` (never ``from config import NAME``)
so that tests can monkeypatch paths at runtime.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Folders
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Files
RAW_FILE = DATA_RAW / "sales.csv"
DAILY_FILE = DATA_PROCESSED / "daily_sales.csv"
MODEL_FILE = MODELS_DIR / "best_model.joblib"
METRICS_FILE = REPORTS_DIR / "metrics.json"
FORECAST_FILE = REPORTS_DIR / "forecast.csv"

# Modelling
TARGET = "revenue"
LAGS = (1, 7, 14, 28, 364)   # 364 = same weekday one year ago (captures yearly seasonality)
ROLL_WINDOWS = (7, 28)
MAX_LOOKBACK = max(max(LAGS), max(ROLL_WINDOWS))
TEST_DAYS = 120            # hold-out period at the end of the series
BACKTEST_HORIZON = 30      # days forecast per backtest window
BACKTEST_ORIGINS = 3       # number of rolling-origin windows inside the hold-out
DEFAULT_HORIZON = 90       # days forecast by the pipeline
RANDOM_STATE = 42
