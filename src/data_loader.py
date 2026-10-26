"""Load, validate, clean and aggregate transactional sales data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config

REQUIRED_COLUMNS = ["date", "store", "category", "units", "revenue"]


def load_raw(path=None) -> pd.DataFrame:
    """Read the raw sales CSV and return a cleaned DataFrame.

    Required columns: date, store, category, units, revenue.
    Optional columns: unit_price, promo (0/1). They are derived/filled when missing.
    """
    path = Path(path or config.RAW_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"Sales data not found at {path}. Run `python -m src.pipeline --regenerate` "
            "to create sample data, or place your own CSV there."
        )

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Sales CSV is missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "store", "category"]).drop_duplicates()

    df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(0).clip(lower=0)
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

    if "unit_price" not in df.columns:
        df["unit_price"] = df["revenue"] / df["units"].replace(0, np.nan)
    df["revenue"] = df["revenue"].fillna(df["units"] * df["unit_price"]).fillna(0).clip(lower=0)

    if "promo" not in df.columns:
        df["promo"] = 0
    df["promo"] = pd.to_numeric(df["promo"], errors="coerce").fillna(0).astype(int).clip(0, 1)

    return df.sort_values("date").reset_index(drop=True)


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per calendar day: revenue, units, promo_share.

    Missing days are inserted and linearly interpolated so lag features stay aligned.
    """
    daily = df.groupby("date").agg(
        revenue=("revenue", "sum"),
        units=("units", "sum"),
        promo_share=("promo", "mean"),
    )
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index)
    daily["revenue"] = daily["revenue"].interpolate(limit_direction="both")
    daily["units"] = daily["units"].interpolate(limit_direction="both")
    daily["promo_share"] = daily["promo_share"].fillna(0.0)
    daily.index.name = "date"
    return daily
