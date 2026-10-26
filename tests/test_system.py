"""Tests. Run with:  pytest -q"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import config
from src.data_generator import generate_sales_data
from src.data_loader import aggregate_daily, load_raw
from src.features import build_features, feature_columns, row_features
from src.pipeline import run


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """Run the whole pipeline on a small dataset inside a temp folder."""
    base = tmp_path_factory.mktemp("work")
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "DATA_RAW", base / "data" / "raw")
    mp.setattr(config, "DATA_PROCESSED", base / "data" / "processed")
    mp.setattr(config, "MODELS_DIR", base / "models")
    mp.setattr(config, "REPORTS_DIR", base / "reports")
    mp.setattr(config, "RAW_FILE", base / "data" / "raw" / "sales.csv")
    mp.setattr(config, "DAILY_FILE", base / "data" / "processed" / "daily_sales.csv")
    mp.setattr(config, "MODEL_FILE", base / "models" / "best_model.joblib")
    mp.setattr(config, "METRICS_FILE", base / "reports" / "metrics.json")
    mp.setattr(config, "FORECAST_FILE", base / "reports" / "forecast.csv")
    result = run(regenerate=True, horizon=30, plots=False, start="2023-01-01", end="2024-12-31")
    yield base, result
    mp.undo()


def test_generator_schema():
    df = generate_sales_data(start="2024-01-01", end="2024-01-31")
    assert {"date", "store", "category", "units", "unit_price", "promo", "revenue"} <= set(df.columns)
    assert (df["revenue"] >= 0).all()
    assert len(df) == 31 * 5 * 5


def test_loader_rejects_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2024-01-01"], "store": ["A"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_raw(bad)


def test_aggregate_fills_missing_days():
    df = generate_sales_data(start="2024-01-01", end="2024-03-31")
    df = df[df["date"] != "2024-02-10"]
    daily = aggregate_daily(df)
    assert len(daily) == 91
    assert daily["revenue"].notna().all()


def test_row_features_match_vectorised_features():
    # needs > 364 days because of the year-ago lag feature
    df = generate_sales_data(start="2022-01-01", end="2023-12-31")
    daily = aggregate_daily(df)
    feat = build_features(daily)
    cols = feature_columns()
    date = feat.index[60]
    history = daily["revenue"][daily.index < date]
    row = row_features(history, date, daily.loc[date, "promo_share"])
    np.testing.assert_allclose(row[cols].to_numpy(float)[0], feat.loc[date, cols].to_numpy(float))


def test_pipeline_outputs(workspace):
    base, result = workspace
    assert config.MODEL_FILE.exists() and config.METRICS_FILE.exists() and config.FORECAST_FILE.exists()
    fc = result["forecast"]
    assert len(fc) == 30
    assert fc["forecast"].notna().all() and (fc["forecast"] >= 0).all()
    assert (fc["lower"] <= fc["forecast"]).all() and (fc["forecast"] <= fc["upper"]).all()
    assert fc["date"].iloc[0] == result["daily"].index[-1] + pd.Timedelta(days=1)


def test_model_beats_reasonable_error(workspace):
    _, result = workspace
    best = result["metrics"]["models"][result["metrics"]["best_model"]]["recursive"]
    assert best["mape"] < 25
    assert result["metrics"]["best_model"] != "SeasonalNaive"


def test_api_endpoints(workspace):
    from api.main import app

    client = TestClient(app)
    assert client.get("/health").json()["model_ready"] is True
    kpis = client.get("/kpis").json()
    assert kpis["total_revenue"] > 0
    body = client.get("/forecast", params={"days": 14}).json()
    assert len(body["forecast"]) == 14
    assert client.get("/forecast", params={"days": 0}).status_code == 422
    assert client.get("/analytics/category").status_code == 200
    assert client.get("/analytics/bogus").status_code == 422
    assert len(client.get("/history", params={"days": 10}).json()) == 10
    assert "best_model" in client.get("/metrics").json()
