"""Static report charts (PNG) written to reports/."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_forecast(history: pd.Series, fc: pd.DataFrame, path, lookback: int = 180) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    h = history.tail(lookback)
    ax.plot(h.index, h.values, label="Actual", color="#1f77b4")
    ax.plot(fc["date"], fc["forecast"], label="Forecast", color="#d62728")
    ax.fill_between(fc["date"], fc["lower"], fc["upper"], color="#d62728", alpha=0.15,
                    label="Approx. 95% interval")
    ax.set(title="Daily revenue: recent actuals and forecast", ylabel="Revenue (INR)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_monthly(monthly: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(monthly["month"], monthly["revenue"], width=20, color="#2a9d8f")
    ax.set(title="Monthly revenue", ylabel="Revenue (INR)")
    ax.grid(axis="y", alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_model_comparison(metrics: dict, path) -> None:
    names = list(metrics["models"].keys())
    rmse = [metrics["models"][n]["recursive"]["rmse"] for n in names]
    colors = ["#d62728" if n == metrics["best_model"] else "#999999" for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, rmse, color=colors)
    ax.set(title="Recursive 30-day backtest RMSE (lower is better)", ylabel="RMSE (INR)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
