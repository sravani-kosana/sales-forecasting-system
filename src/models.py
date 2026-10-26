"""Model training, honest evaluation and selection.

Every candidate is scored two ways on a chronological hold-out (last TEST_DAYS days):
* one-step-ahead  - uses true lags (optimistic)
* recursive       - rolling-origin 30-day forecasts that feed predictions back in (realistic)
The best ML model is chosen by recursive RMSE, refit on all data and saved.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.features import build_features, feature_columns
from src.forecast import recursive_forecast

BASELINE_NAME = "SeasonalNaive"


class SeasonalNaiveModel:
    """Baseline: predict the value from the same weekday last week."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.asarray(X["lag_7"], dtype=float)


def get_candidates() -> dict:
    return {
        BASELINE_NAME: SeasonalNaiveModel(),
        "Ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, n_jobs=1, random_state=config.RANDOM_STATE
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.8,
            random_state=config.RANDOM_STATE,
        ),
    }


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    mask = y_true > 0
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "mae": round(float(np.abs(err).mean()), 2),
        "rmse": round(float(np.sqrt((err ** 2).mean())), 2),
        "mape": round(float((np.abs(err[mask]) / y_true[mask]).mean() * 100), 2),
        "r2": round(1 - float((err ** 2).sum()) / ss_tot, 4) if ss_tot > 0 else 0.0,
    }


def _recursive_backtest(model, daily: pd.DataFrame, test_start) -> dict:
    series = daily[config.TARGET]
    y_true, y_pred = [], []
    for k in range(config.BACKTEST_ORIGINS):
        origin = test_start + pd.Timedelta(days=k * config.BACKTEST_HORIZON)
        history = series[series.index < origin]
        dates = pd.date_range(origin, periods=config.BACKTEST_HORIZON, freq="D")
        promo = daily["promo_share"].reindex(dates).fillna(0).to_numpy()
        pred = recursive_forecast(model, history, dates, promo)
        y_true.append(series.reindex(dates).to_numpy())
        y_pred.append(pred.to_numpy())
    return regression_metrics(np.concatenate(y_true), np.concatenate(y_pred))


def _importance(model, cols) -> dict:
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "named_steps") and hasattr(model.named_steps["model"], "coef_"):
        vals = np.abs(model.named_steps["model"].coef_)
        vals = vals / vals.sum()
    else:
        return {}
    pairs = sorted(zip(cols, vals), key=lambda p: -p[1])
    return {c: round(float(v), 4) for c, v in pairs}


def train(daily: pd.DataFrame, save: bool = True) -> tuple[dict, dict]:
    """Train all candidates, pick the best, refit on everything. Returns (bundle, metrics)."""
    feat = build_features(daily)
    if len(feat) < config.TEST_DAYS + 180:
        raise ValueError(
            f"Not enough history: need at least {config.TEST_DAYS + 180 + config.MAX_LOOKBACK} "
            f"days, got {len(daily)}."
        )
    cols = feature_columns()
    target = config.TARGET
    train_df = feat.iloc[: -config.TEST_DAYS]
    test_df = feat.iloc[-config.TEST_DAYS:]

    results: dict[str, dict] = {}
    for name, model in get_candidates().items():
        model.fit(train_df[cols], train_df[target])
        results[name] = {
            "one_step": regression_metrics(test_df[target], model.predict(test_df[cols])),
            "recursive": _recursive_backtest(model, daily, test_df.index[0]),
        }

    ml_results = {k: v for k, v in results.items() if k != BASELINE_NAME}
    best_name = min(ml_results, key=lambda k: ml_results[k]["recursive"]["rmse"])

    final_model = get_candidates()[best_name]
    final_model.fit(feat[cols], feat[target])

    bundle = {
        "model": final_model,
        "model_name": best_name,
        "feature_columns": cols,
        "rmse": results[best_name]["recursive"]["rmse"],
        "trained_until": str(daily.index[-1].date()),
        "default_promo_share": float(daily["promo_share"].tail(365).mean()),
    }
    metrics = {
        "best_model": best_name,
        "trained_until": bundle["trained_until"],
        "train_rows": int(len(train_df)),
        "test_days": config.TEST_DAYS,
        "backtest": {
            "horizon_days": config.BACKTEST_HORIZON,
            "origins": config.BACKTEST_ORIGINS,
        },
        "models": results,
        "feature_importance": _importance(final_model, cols),
    }

    if save:
        config.MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        config.METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, config.MODEL_FILE)
        config.METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    return bundle, metrics
