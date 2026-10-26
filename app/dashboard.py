"""Streamlit dashboard.  Start with:  streamlit run app/dashboard.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import analytics, config  # noqa: E402
from src.data_loader import aggregate_daily, load_raw  # noqa: E402
from src.forecast import generate_forecast, load_bundle  # noqa: E402

st.set_page_config(page_title="Sales Forecasting & Analytics", page_icon="📈", layout="wide")


def inr(value) -> str:
    """Format a number in Indian units (Lakh / Crore)."""
    if value is None:
        return "–"
    v = abs(float(value))
    sign = "-" if value < 0 else ""
    if v >= 1e7:
        return f"{sign}₹{v / 1e7:.2f} Cr"
    if v >= 1e5:
        return f"{sign}₹{v / 1e5:.2f} L"
    return f"{sign}₹{v:,.0f}"


def delta(value, label: str):
    return None if value is None else f"{value:+.1f}% {label}"


@st.cache_data(show_spinner="Loading sales data…")
def load_data():
    raw = load_raw()
    return raw, aggregate_daily(raw)


@st.cache_resource
def get_bundle():
    return load_bundle()


@st.cache_data
def get_metrics() -> dict:
    return json.loads(config.METRICS_FILE.read_text()) if config.METRICS_FILE.exists() else {}


@st.cache_data(show_spinner="Forecasting…")
def cached_forecast(days: int, promo_share: float) -> pd.DataFrame:
    _, daily_all = load_data()
    return generate_forecast(get_bundle(), daily_all["revenue"], days, promo_share)


try:
    raw_all, daily_all = load_data()
    bundle = get_bundle()
except FileNotFoundError as exc:
    st.error(f"{exc}")
    st.info("Run `python -m src.pipeline --regenerate` in the project folder, then refresh this page.")
    st.stop()

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
min_d, max_d = raw_all["date"].min().date(), raw_all["date"].max().date()
picked = st.sidebar.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
if not isinstance(picked, (tuple, list)) or len(picked) != 2:
    st.info("Select both a start and an end date.")
    st.stop()
stores = st.sidebar.multiselect("Stores", sorted(raw_all["store"].unique()), default=sorted(raw_all["store"].unique()))
cats = st.sidebar.multiselect("Categories", sorted(raw_all["category"].unique()), default=sorted(raw_all["category"].unique()))

mask = (
    (raw_all["date"] >= pd.Timestamp(picked[0]))
    & (raw_all["date"] <= pd.Timestamp(picked[1]))
    & raw_all["store"].isin(stores)
    & raw_all["category"].isin(cats)
)
raw = raw_all[mask]
if raw.empty:
    st.warning("No data for the selected filters.")
    st.stop()
daily = aggregate_daily(raw)
kpis = analytics.compute_kpis(raw, daily)

st.title("📈 Predictive Sales Forecasting & Business Analytics")
st.caption(f"Data: {kpis['date_from']} → {kpis['date_to']}  |  Forecast model: {bundle['model_name']} "
           f"(trained until {bundle['trained_until']})")

tab_overview, tab_trends, tab_segments, tab_forecast, tab_model = st.tabs(
    ["Overview", "Trends", "Segments", "Forecast", "Model"]
)

# ---------------- Overview ----------------
with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total revenue", inr(kpis["total_revenue"]))
    c2.metric("Units sold", f"{kpis['total_units']:,}")
    c3.metric("Avg daily revenue", inr(kpis["avg_daily_revenue"]))
    c4.metric("Last 30 days", inr(kpis["last_30d_revenue"]), delta(kpis["mom_growth_pct"], "vs prev 30d"))
    c5.metric("YoY growth", "–" if kpis["yoy_growth_pct"] is None else f"{kpis['yoy_growth_pct']:+.1f}%")
    st.write(f"**Top store:** {kpis['top_store']}  ·  **Top category:** {kpis['top_category']}  ·  "
             f"**Revenue from promotions:** {kpis['promo_revenue_share_pct']}%")

    monthly = analytics.monthly_revenue(raw)
    fig = px.bar(monthly, x="month", y="revenue", title="Monthly revenue")
    st.plotly_chart(fig)

# ---------------- Trends ----------------
with tab_trends:
    d = daily.reset_index()
    d["MA 7d"] = d["revenue"].rolling(7).mean()
    d["MA 28d"] = d["revenue"].rolling(28).mean()
    fig = px.line(d, x="date", y=["revenue", "MA 7d", "MA 28d"], title="Daily revenue with moving averages")
    st.plotly_chart(fig)
    left, right = st.columns(2)
    wd = analytics.weekday_pattern(daily)
    left.plotly_chart(px.bar(wd, x="weekday", y="avg_revenue", title="Average revenue by weekday"))
    right.write("**Monthly table**")
    right.dataframe(monthly.assign(month=monthly["month"].dt.strftime("%Y-%m")))

# ---------------- Segments ----------------
with tab_segments:
    left, right = st.columns(2)
    by_store = analytics.revenue_by(raw, "store")
    by_cat = analytics.revenue_by(raw, "category")
    left.plotly_chart(px.bar(by_store, x="store", y="revenue", title="Revenue by store"))
    right.plotly_chart(px.bar(by_cat, x="category", y="revenue", title="Revenue by category"))
    st.write("**ABC analysis by category** (A = top 70% of revenue, B = next 20%, C = rest)")
    st.dataframe(by_cat)
    matrix = analytics.category_store_matrix(raw)
    st.plotly_chart(px.imshow(matrix, text_auto=".2s", aspect="auto", title="Revenue: store × category"))
    st.write("**Promotion effect** (average units per store-day)")
    st.dataframe(analytics.promo_effect(raw))

# ---------------- Forecast ----------------
with tab_forecast:
    st.caption("Forecast covers all stores and categories combined (it ignores the sidebar filters).")
    c1, c2 = st.columns(2)
    horizon = c1.slider("Forecast horizon (days)", 7, 180, 90)
    promo_pct = c2.slider("Planned promotion share (%)", 0, 50, int(round(bundle["default_promo_share"] * 100)))
    fc = cached_forecast(horizon, promo_pct / 100)

    hist = daily_all["revenue"].tail(180)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Actual", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["lower"], fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(214,39,40,0.15)", name="Approx. 95% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast"], name="Forecast", line=dict(color="#d62728")))
    fig.update_layout(title="Daily revenue forecast", yaxis_title="Revenue (INR)")
    st.plotly_chart(fig)

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Forecast revenue (next {min(30, horizon)} days)", inr(fc["forecast"].head(30).sum()))
    m2.metric("Avg forecast per day", inr(fc["forecast"].mean()))
    peak = fc.loc[fc["forecast"].idxmax()]
    m3.metric("Peak day", peak["date"].strftime("%d %b %Y"), inr(peak["forecast"]), delta_color="off")
    st.dataframe(fc.assign(date=fc["date"].dt.strftime("%Y-%m-%d")))
    st.download_button("Download forecast CSV", fc.to_csv(index=False), "forecast.csv", "text/csv")

# ---------------- Model ----------------
with tab_model:
    metrics = get_metrics()
    if not metrics:
        st.info("No metrics found. Run the pipeline first.")
    else:
        st.success(f"Selected model: **{metrics['best_model']}** (lowest recursive 30-day RMSE)")
        rows = []
        for name, res in metrics["models"].items():
            rows.append({
                "Model": name,
                "One-step MAPE %": res["one_step"]["mape"],
                "One-step RMSE": res["one_step"]["rmse"],
                "Recursive MAPE %": res["recursive"]["mape"],
                "Recursive RMSE": res["recursive"]["rmse"],
                "Recursive R²": res["recursive"]["r2"],
            })
        st.dataframe(pd.DataFrame(rows))
        st.caption("Recursive metrics (rolling 30-day forecasts on the hold-out) are the realistic ones; "
                   "one-step metrics use true yesterday values and are optimistic.")
        imp = pd.Series(metrics["feature_importance"]).head(12).sort_values()
        if not imp.empty:
            st.plotly_chart(px.bar(imp, orientation="h", title="Top feature importances",
                                   labels={"value": "importance", "index": "feature"}))
