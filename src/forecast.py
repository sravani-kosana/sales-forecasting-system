"""Multi-step forecasting with a trained model bundle."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.features import row_features


def load_bundle(path=None) -> dict:
    """Load the saved model bundle (model + metadata)."""
    path = Path(path or config.MODEL_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run `python -m src.pipeline` first."
        )
    return joblib.load(path)


def recursive_forecast(model, history: pd.Series, future_dates, promo_share) -> pd.Series:
    """Predict day by day, feeding each prediction back in as the next day's lag."""
    future_dates = pd.DatetimeIndex(future_dates)
    if np.isscalar(promo_share):
        promo = np.full(len(future_dates), float(promo_share))
    else:
        promo = np.asarray(promo_share, dtype=float)
    if len(promo) != len(future_dates):
        raise ValueError("promo_share must be a scalar or match the number of forecast days")

    values = list(np.asarray(history, dtype=float))
    preds = []
    for date, p in zip(future_dates, promo):
        x = row_features(values, date, p)
        y_hat = float(max(model.predict(x)[0], 0.0))
        preds.append(y_hat)
        values.append(y_hat)
    return pd.Series(preds, index=future_dates, name="forecast")


def generate_forecast(bundle: dict, history: pd.Series, days: int, promo_share=None) -> pd.DataFrame:
    """Forecast ``days`` days after the last date in ``history``.

    Returns columns: date, forecast, lower, upper. The interval is an approximation:
    forecast +/- 1.96 * RMSE of the recursive backtest.
    """
    if days < 1:
        raise ValueError("days must be >= 1")
    promo = bundle["default_promo_share"] if promo_share is None else promo_share
    future = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=days, freq="D")
    pred = recursive_forecast(bundle["model"], history, future, promo)

    margin = 1.96 * float(bundle["rmse"])
    out = pd.DataFrame({"date": future, "forecast": pred.to_numpy()})
    out["lower"] = (out["forecast"] - margin).clip(lower=0)
    out["upper"] = out["forecast"] + margin
    num_cols = ["forecast", "lower", "upper"]
    out[num_cols] = out[num_cols].round(2)
    return out
