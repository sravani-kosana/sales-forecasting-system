"""Feature engineering.

Two code paths produce the *same* features:
* ``build_features``  - vectorised, used for training
* ``row_features``    - one row at a time, used for recursive multi-step forecasting
A unit test asserts they agree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

CALENDAR_COLUMNS = [
    "dayofweek", "month", "day", "dayofyear", "weekofyear", "quarter",
    "is_weekend", "is_month_start", "is_month_end",
    "doy_sin", "doy_cos", "dow_sin", "dow_cos", "t",
]
_EPOCH = pd.Timestamp("2020-01-01")


def calendar_features(index) -> pd.DataFrame:
    idx = pd.DatetimeIndex(index)
    f = pd.DataFrame(index=idx)
    f["dayofweek"] = idx.dayofweek
    f["month"] = idx.month
    f["day"] = idx.day
    f["dayofyear"] = idx.dayofyear
    f["weekofyear"] = idx.isocalendar().week.to_numpy().astype(int)
    f["quarter"] = idx.quarter
    f["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    f["is_month_start"] = idx.is_month_start.astype(int)
    f["is_month_end"] = idx.is_month_end.astype(int)
    f["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    f["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    f["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    f["t"] = np.asarray((idx - _EPOCH).days)
    return f


def feature_columns() -> list[str]:
    cols = list(CALENDAR_COLUMNS) + ["promo_share"]
    cols += [f"lag_{lag}" for lag in config.LAGS]
    cols += [f"roll_mean_{w}" for w in config.ROLL_WINDOWS]
    return cols


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Training matrix from the daily table (rows without full lag history are dropped)."""
    df = calendar_features(daily.index)
    df["promo_share"] = daily["promo_share"].to_numpy()
    y = daily[config.TARGET]
    for lag in config.LAGS:
        df[f"lag_{lag}"] = y.shift(lag).to_numpy()
    for w in config.ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = y.shift(1).rolling(w).mean().to_numpy()
    df[config.TARGET] = y.to_numpy()
    return df.dropna().copy()


def row_features(history, date, promo_share: float) -> pd.DataFrame:
    """Features for a single future ``date`` given the revenue ``history`` before it."""
    h = np.asarray(history, dtype=float)
    if len(h) < config.MAX_LOOKBACK:
        raise ValueError(f"Need at least {config.MAX_LOOKBACK} days of history, got {len(h)}")
    row = calendar_features([date]).iloc[0].to_dict()
    row["promo_share"] = float(promo_share)
    for lag in config.LAGS:
        row[f"lag_{lag}"] = h[-lag]
    for w in config.ROLL_WINDOWS:
        row[f"roll_mean_{w}"] = h[-w:].mean()
    return pd.DataFrame([row])[feature_columns()]
