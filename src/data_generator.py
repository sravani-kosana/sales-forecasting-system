"""Synthetic retail sales generator.

Used when no real dataset is supplied. Produces one row per (date, store, category)
with trend, weekly seasonality, yearly seasonality, festive-season uplift and promotions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config

# store -> demand multiplier
STORES = {
    "Hyderabad": 1.15,
    "Mumbai": 1.30,
    "Delhi": 1.25,
    "Bengaluru": 1.10,
    "Chennai": 0.90,
}

# category -> (base units per day, base unit price in INR)
CATEGORIES = {
    "Electronics": (6, 14000.0),
    "Clothing": (25, 1800.0),
    "Groceries": (80, 450.0),
    "Furniture": (3, 9500.0),
    "Sports": (12, 2600.0),
}


def generate_sales_data(
    start: str = "2022-01-01",
    end: str = "2026-09-30",
    seed: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Return a DataFrame with columns: date, store, category, units, unit_price, promo, revenue."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    t = np.arange(n)
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()
    month = dates.month.to_numpy()

    trend = 1 + 0.00035 * t
    weekend = np.where(dow >= 5, 1.28, 1.0)
    yearly = 1 + 0.18 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    festive = np.where(np.isin(month, [10, 11]), 1.30, 1.0) * np.where(month == 12, 1.15, 1.0)
    shared = trend * weekend * yearly * festive

    frames = []
    for store, store_factor in STORES.items():
        for category, (base_units, price) in CATEGORIES.items():
            promo = rng.random(n) < 0.12
            lam = base_units * store_factor * shared * np.where(promo, 1.35, 1.0)
            units = rng.poisson(lam)
            unit_price = price * np.where(promo, 0.88, 1.0) * (1 + rng.normal(0, 0.015, n))
            frames.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "store": store,
                        "category": category,
                        "units": units,
                        "unit_price": unit_price.round(2),
                        "promo": promo.astype(int),
                        "revenue": (units * unit_price).round(2),
                    }
                )
            )

    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["date", "store", "category"]).reset_index(drop=True)


def save_sales_data(path=None, **kwargs) -> Path:
    """Generate the dataset and write it to CSV (defaults to config.RAW_FILE)."""
    path = Path(path or config.RAW_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_sales_data(**kwargs).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    out = save_sales_data()
    print(f"Wrote {out}")
