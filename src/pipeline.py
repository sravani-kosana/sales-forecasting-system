"""End-to-end pipeline: data -> clean -> train -> evaluate -> forecast -> reports.

Usage:
    python -m src.pipeline --regenerate          # create sample data, then run everything
    python -m src.pipeline                       # use existing data/raw/sales.csv
    python -m src.pipeline --horizon 120 --no-plots
"""
from __future__ import annotations

import argparse

from src import analytics, config
from src.data_generator import save_sales_data
from src.data_loader import aggregate_daily, load_raw
from src.forecast import generate_forecast
from src.models import train


def run(
    regenerate: bool = False,
    horizon: int = config.DEFAULT_HORIZON,
    plots: bool = True,
    start: str = "2022-01-01",
    end: str = "2026-09-30",
) -> dict:
    for folder in (config.DATA_RAW, config.DATA_PROCESSED, config.MODELS_DIR, config.REPORTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    if regenerate or not config.RAW_FILE.exists():
        path = save_sales_data(start=start, end=end)
        print(f"[1/5] Generated sample data -> {path}")
    else:
        print(f"[1/5] Using existing data -> {config.RAW_FILE}")

    raw = load_raw()
    daily = aggregate_daily(raw)
    daily.to_csv(config.DAILY_FILE)
    print(f"[2/5] Cleaned {len(raw):,} rows -> {len(daily):,} daily records "
          f"({daily.index.min().date()} to {daily.index.max().date()})")

    bundle, metrics = train(daily)
    best = metrics["models"][metrics["best_model"]]
    print(f"[3/5] Trained models. Best: {metrics['best_model']} "
          f"(30-day recursive MAPE {best['recursive']['mape']}%, R2 {best['recursive']['r2']})")

    forecast = generate_forecast(bundle, daily[config.TARGET], horizon)
    forecast.to_csv(config.FORECAST_FILE, index=False)
    print(f"[4/5] Forecast {horizon} days ahead -> {config.FORECAST_FILE}")

    if plots:
        from src import visualize

        visualize.plot_forecast(daily[config.TARGET], forecast, config.REPORTS_DIR / "forecast.png")
        visualize.plot_monthly(analytics.monthly_revenue(raw), config.REPORTS_DIR / "monthly_revenue.png")
        visualize.plot_model_comparison(metrics, config.REPORTS_DIR / "model_comparison.png")
        print(f"[5/5] Saved charts -> {config.REPORTS_DIR}")
    else:
        print("[5/5] Skipped charts")

    return {"bundle": bundle, "metrics": metrics, "forecast": forecast, "daily": daily}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the sales forecasting pipeline.")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate the sample dataset")
    parser.add_argument("--horizon", type=int, default=config.DEFAULT_HORIZON, help="Forecast days")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG chart generation")
    args = parser.parse_args(argv)
    run(regenerate=args.regenerate, horizon=args.horizon, plots=not args.no_plots)


if __name__ == "__main__":
    main()
