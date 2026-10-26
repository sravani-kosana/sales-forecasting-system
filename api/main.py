"""REST API.  Start with:  uvicorn api.main:app --reload   (docs at /docs)"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from src import analytics, config
from src.data_loader import aggregate_daily, load_raw
from src.forecast import generate_forecast, load_bundle

app = FastAPI(
    title="Predictive Sales Forecasting API",
    version="1.0.0",
    description="Sales KPIs, business analytics and ML-based revenue forecasts.",
)


# ---------- helpers ----------
@lru_cache(maxsize=4)
def _load_data(path: str, mtime: float):
    raw = load_raw(path)
    return raw, aggregate_daily(raw)


@lru_cache(maxsize=4)
def _load_model(path: str, mtime: float):
    return load_bundle(path)


def _data():
    path = config.RAW_FILE
    if not path.exists():
        raise HTTPException(503, "Sales data not found. Run `python -m src.pipeline --regenerate`.")
    return _load_data(str(path), path.stat().st_mtime)


def _bundle():
    path = config.MODEL_FILE
    if not path.exists():
        raise HTTPException(503, "Model not trained. Run `python -m src.pipeline`.")
    return _load_model(str(path), path.stat().st_mtime)


def _records(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    for col in df.select_dtypes(include=["datetime", "datetimetz"]).columns:
        df[col] = df[col].dt.strftime("%Y-%m-%d")
    return json.loads(df.to_json(orient="records"))


# ---------- endpoints ----------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "data_ready": config.RAW_FILE.exists(),
        "model_ready": config.MODEL_FILE.exists(),
    }


@app.get("/kpis")
def kpis():
    raw, daily = _data()
    return analytics.compute_kpis(raw, daily)


@app.get("/metrics")
def metrics():
    if not config.METRICS_FILE.exists():
        raise HTTPException(503, "Metrics not found. Run `python -m src.pipeline`.")
    return json.loads(config.METRICS_FILE.read_text())


@app.get("/history")
def history(days: int = Query(90, ge=1, le=3650)):
    _, daily = _data()
    tail = daily.tail(days).reset_index()
    return _records(tail[["date", "revenue", "units"]])


@app.get("/forecast")
def forecast(
    days: int = Query(30, ge=1, le=365, description="Days to forecast after the last data date"),
    promo_share: Optional[float] = Query(
        None, ge=0, le=1, description="Planned share of store-category lines on promotion (0-1)"
    ),
):
    _, daily = _data()
    bundle = _bundle()
    fc = generate_forecast(bundle, daily["revenue"], days, promo_share)
    return {
        "model": bundle["model_name"],
        "trained_until": bundle["trained_until"],
        "days": days,
        "promo_share": bundle["default_promo_share"] if promo_share is None else promo_share,
        "total_forecast_revenue": round(float(fc["forecast"].sum()), 2),
        "forecast": _records(fc),
    }


@app.get("/analytics/{dimension}")
def analytics_view(dimension: Literal["monthly", "category", "store", "weekday", "promo"]):
    raw, daily = _data()
    if dimension == "monthly":
        df = analytics.monthly_revenue(raw)
    elif dimension == "category":
        df = analytics.revenue_by(raw, "category")
    elif dimension == "store":
        df = analytics.revenue_by(raw, "store")
    elif dimension == "weekday":
        df = analytics.weekday_pattern(daily)
    else:
        df = analytics.promo_effect(raw)
    return _records(df)
