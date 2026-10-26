"""Business analytics on the raw (transaction-level) sales table."""
from __future__ import annotations

import numpy as np
import pandas as pd

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _pct_change(current: float, previous: float):
    return None if not previous else round(float((current / previous - 1) * 100), 2)


def compute_kpis(raw: pd.DataFrame, daily: pd.DataFrame) -> dict:
    """Headline KPIs. Growth values are None when there is not enough history."""
    if raw.empty or daily.empty:
        raise ValueError("No data available to compute KPIs.")
    rev = daily["revenue"]
    total = float(raw["revenue"].sum())
    by_store = raw.groupby("store")["revenue"].sum()
    by_cat = raw.groupby("category")["revenue"].sum()
    promo_rev = float(raw.loc[raw["promo"] == 1, "revenue"].sum())

    return {
        "date_from": str(rev.index.min().date()),
        "date_to": str(rev.index.max().date()),
        "total_revenue": round(total, 2),
        "total_units": int(raw["units"].sum()),
        "avg_daily_revenue": round(float(rev.mean()), 2),
        "last_30d_revenue": round(float(rev.tail(30).sum()), 2),
        "mom_growth_pct": _pct_change(rev.tail(30).sum(), rev.iloc[-60:-30].sum()) if len(rev) >= 60 else None,
        "yoy_growth_pct": _pct_change(rev.tail(365).sum(), rev.iloc[-730:-365].sum()) if len(rev) >= 730 else None,
        "top_store": str(by_store.idxmax()),
        "top_category": str(by_cat.idxmax()),
        "promo_revenue_share_pct": round(promo_rev / total * 100, 2) if total else 0.0,
    }


def monthly_revenue(raw: pd.DataFrame) -> pd.DataFrame:
    m = raw.groupby(raw["date"].dt.to_period("M")).agg(
        revenue=("revenue", "sum"), units=("units", "sum")
    )
    m.index = m.index.to_timestamp()
    m.index.name = "month"
    m["mom_growth_pct"] = (m["revenue"].pct_change() * 100).round(2)
    m["revenue"] = m["revenue"].round(2)
    return m.reset_index()


def revenue_by(raw: pd.DataFrame, column: str) -> pd.DataFrame:
    """Revenue/units by ``column`` with share, cumulative share and an ABC class (A=top 70%, B=next 20%)."""
    g = raw.groupby(column).agg(revenue=("revenue", "sum"), units=("units", "sum"))
    g = g.sort_values("revenue", ascending=False)
    g["share_pct"] = (g["revenue"] / g["revenue"].sum() * 100).round(2)
    g["cum_share_pct"] = g["share_pct"].cumsum().round(2)
    before = g["cum_share_pct"] - g["share_pct"]
    g["abc_class"] = np.where(before < 70, "A", np.where(before < 90, "B", "C"))
    g["revenue"] = g["revenue"].round(2)
    return g.reset_index()


def weekday_pattern(daily: pd.DataFrame) -> pd.DataFrame:
    g = daily.groupby(daily.index.dayofweek)["revenue"].mean()
    return pd.DataFrame(
        {"weekday": [WEEKDAYS[i] for i in g.index], "avg_revenue": g.round(2).to_numpy()}
    )


def promo_effect(raw: pd.DataFrame) -> pd.DataFrame:
    """Average units sold per store-day with and without a promotion, by category."""
    g = raw.groupby(["category", "promo"])["units"].mean().unstack("promo")
    if 0 not in g.columns or 1 not in g.columns:
        return pd.DataFrame(columns=["category", "no_promo_avg_units", "promo_avg_units", "lift_pct"])
    g = g[[0, 1]]
    g.columns = ["no_promo_avg_units", "promo_avg_units"]
    g["lift_pct"] = ((g["promo_avg_units"] / g["no_promo_avg_units"] - 1) * 100).round(2)
    return g.round(2).reset_index()


def category_store_matrix(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.pivot_table(
        index="store", columns="category", values="revenue", aggfunc="sum"
    ).round(0)
