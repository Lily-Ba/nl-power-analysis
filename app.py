"""Dutch power market dashboard: prices, negative prices, forecasts, procurement."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA = Path("data/processed")
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"

st.set_page_config(page_title="Dutch Power Market", page_icon="⚡", layout="wide")


@st.cache_data
def load_data():
    hourly = pd.read_parquet(DATA / "hourly.parquet")
    preds = pd.read_parquet(DATA / "predictions_2025.parquet")
    importance = pd.read_parquet(DATA / "feature_importance.parquet")["importance"]
    metrics = json.loads((DATA / "metrics.json").read_text())
    return hourly, preds, importance, metrics


def style(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.1, x=0, title_text=""), hovermode="x unified")
    return fig


hourly, preds, importance, metrics = load_data()
yearly = hourly.groupby("year").agg(avg_price=("price", "mean"),
                                    negative_hours=("is_negative", "sum"))

st.title("⚡ Dutch Power Market Dashboard")
st.caption("Netherlands day-ahead electricity prices 2019–2025 · Source: ENTSO-E Transparency Platform")

tabs = st.tabs(["Market overview", "Negative prices", "Price forecast",
                "Negative-price alert", "Procurement simulator"])

# --- 1. Market overview -------------------------------------------------------
with tabs[0]:
    latest = yearly.index.max()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Average price {latest}", f"€{yearly.loc[latest, 'avg_price']:.0f}/MWh")
    c2.metric("Crisis peak year (2022)", f"€{yearly.loc[2022, 'avg_price']:.0f}/MWh")
    c3.metric(f"Negative hours {latest}", f"{yearly.loc[latest, 'negative_hours']:.0f}")
    c4.metric("Negative hours 2019", f"{yearly.loc[2019, 'negative_hours']:.0f}")

    monthly = hourly["price"].resample("MS").mean().rename("price").reset_index()
    fig = px.line(monthly, x="timestamp", y="price", color_discrete_sequence=[BLUE],
                  labels={"timestamp": "", "price": "€/MWh"}, title="Monthly average day-ahead price")
    st.plotly_chart(style(fig), width="stretch")

    year = st.select_slider("Year for the daily profile", options=sorted(hourly["year"].unique()),
                            value=latest)
    heat = hourly[hourly["year"] == year].pivot_table(index="hour", columns="month",
                                                      values="price", aggfunc="mean")
    fig = px.imshow(heat, color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                    labels=dict(x="Month", y="Hour of day", color="€/MWh"), aspect="auto",
                    title=f"Average price by hour and month, {year}: the solar 'duck curve'")
    st.plotly_chart(style(fig, 460), width="stretch")

# --- 2. Negative prices ---------------------------------------------------------
with tabs[1]:
    fig = px.bar(yearly.reset_index(), x="year", y="negative_hours", text="negative_hours",
                 color_discrete_sequence=[BLUE], title="Hours with a negative day-ahead price",
                 labels={"year": "", "negative_hours": "Hours"})
    st.plotly_chart(style(fig), width="stretch")

    left, right = st.columns(2)
    recent = hourly[hourly["year"] >= 2023]
    by_hour = recent.groupby("hour")["is_negative"].mean().mul(100).reset_index()
    fig = px.bar(by_hour, x="hour", y="is_negative", color_discrete_sequence=[BLUE],
                 title="Share of hours negative, by hour of day (2023–2025)",
                 labels={"hour": "Hour of day", "is_negative": "% of hours"})
    left.plotly_chart(style(fig), width="stretch")

    if "gen_solar" in recent:
        bins = pd.cut(recent["gen_solar"], bins=10)
        by_solar = recent.groupby(bins, observed=True)["is_negative"].mean().mul(100)
        by_solar = by_solar.reset_index().assign(gen_solar=lambda d: d["gen_solar"].map(
            lambda b: f"{b.left / 1000:.0f}–{b.right / 1000:.0f}"))
        fig = px.bar(by_solar, x="gen_solar", y="is_negative", color_discrete_sequence=[ORANGE],
                     title="More solar output → more negative prices (2023–2025)",
                     labels={"gen_solar": "Solar output (GW)", "is_negative": "% of hours negative"})
        right.plotly_chart(style(fig), width="stretch")

# --- 3. Price forecast ----------------------------------------------------------
with tabs[2]:
    mae = metrics["mae"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Model error (MAE)", f"€{mae['Gradient boosting model']:.1f}/MWh")
    c2.metric("Naive forecast error", f"€{mae['Naive: same hour yesterday']:.1f}/MWh")
    c3.metric("Improvement vs naive", f"{metrics['improvement_vs_naive']:.0%}")
    st.caption("Model trained on 2019–2024 and tested on 2025, a year it never saw. "
               "It uses only information available the day before delivery.")

    start = st.date_input("Week starting", value=pd.Timestamp("2025-05-01"),
                          min_value=preds.index.min().date(), max_value=preds.index.max().date())
    week = preds.loc[str(start):str(pd.Timestamp(start) + pd.Timedelta(days=6))]
    fig = go.Figure([
        go.Scatter(x=week.index, y=week["price"], name="Actual", line=dict(color=BLUE, width=2)),
        go.Scatter(x=week.index, y=week["price_pred"], name="Model forecast",
                   line=dict(color=ORANGE, width=2)),
        go.Scatter(x=week.index, y=week["price_lag_24"], name="Naive (yesterday)",
                   line=dict(color=GREY, width=1.5, dash="dot")),
    ])
    fig.add_hline(y=0, line_width=1, line_color=GREY)
    fig.update_layout(title="Actual vs forecast price", yaxis_title="€/MWh")
    st.plotly_chart(style(fig, 420), width="stretch")

    imp = importance.head(10).sort_values().reset_index()
    imp.columns = ["input", "importance"]
    fig = px.bar(imp, x="importance", y="input", orientation="h", color_discrete_sequence=[BLUE],
                 title="What drives the forecast (error increase when the input is scrambled)",
                 labels={"importance": "€/MWh", "input": ""})
    st.plotly_chart(style(fig), width="stretch")

# --- 4. Negative-price alert ----------------------------------------------------
with tabs[3]:
    threshold = st.slider("Alert threshold (probability)", 0.05, 0.95, 0.5, 0.05,
                          help="Lower = catch more negative hours, but more false alarms.")
    alert = preds["neg_prob"] >= threshold
    actual = preds["is_negative"] == 1
    caught = (alert & actual).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Negative hours in 2025", f"{actual.sum()}")
    c2.metric("Caught by alerts (recall)", f"{caught / max(actual.sum(), 1):.0%}")
    c3.metric("Alerts that were right (precision)", f"{caught / max(alert.sum(), 1):.0%}")

    day = st.date_input("Day", value=pd.Timestamp("2025-05-11"),
                        min_value=preds.index.min().date(), max_value=preds.index.max().date(),
                        key="alert_day")
    d = preds.loc[str(day)]
    fig = go.Figure([
        go.Bar(x=d.index.hour, y=d["neg_prob"] * 100, name="Predicted probability",
               marker_color=[ORANGE if p >= threshold else BLUE for p in d["neg_prob"]]),
    ])
    fig.add_hline(y=threshold * 100, line_dash="dot", line_color=GREY,
                  annotation_text="alert threshold")
    fig.update_layout(title="Probability of a negative price, by hour (orange = alert)",
                      xaxis_title="Hour of day", yaxis_title="%", showlegend=False)
    st.plotly_chart(style(fig), width="stretch")
    st.caption("Actual prices that day: " + ", ".join(
        f"{h}h €{p:.0f}" for h, p in zip(d.index.hour, d["price"])))

# --- 5. Procurement simulator ---------------------------------------------------
with tabs[4]:
    st.markdown("A 10 MW industrial consumer buys power under different strategies. "
                "Cost is shown per MWh consumed, for each calendar year.")
    c1, c2, c3, c4 = st.columns(4)
    profile = c1.radio("Load profile", ["24/7 baseload", "Weekday daytime (08–18)"])
    hedge = c2.slider("Hedge ratio (layered strategy)", 0, 100, 60, 10) / 100
    premium = c3.slider("Forward premium (€/MWh)", 0, 20, 5)
    ppa_price = c4.slider("Solar PPA price (€/MWh)", 30, 90, 55)
    ppa_share = st.slider("Solar PPA volume (% of annual consumption)", 10, 100, 40, 10) / 100

    sim = hourly[["price", "year", "hour", "weekday", "gen_solar"]].dropna().copy()
    if profile == "24/7 baseload":
        sim["load"] = 10.0
    else:
        sim["load"] = np.where((sim["weekday"] < 5) & sim["hour"].between(8, 17), 10.0, 0.0)

    # Fixed price for year Y = previous year's average baseload price + premium
    fixed = yearly["avg_price"].shift(1) + premium

    rows = []
    for year, g in sim.groupby("year"):
        if pd.isna(fixed.get(year)):
            continue
        volume = g["load"].sum()
        spot = (g["load"] * g["price"]).sum() / volume
        ppa_mwh = g["gen_solar"] / g["gen_solar"].sum() * ppa_share * volume
        ppa = (ppa_mwh * ppa_price + (g["load"] - ppa_mwh) * g["price"]).sum() / volume
        rows.append({"year": year, "100% spot": spot, "Fixed price": fixed[year],
                     f"Layered ({hedge:.0%} fixed)": hedge * fixed[year] + (1 - hedge) * spot,
                     "Solar PPA + spot": ppa})
    result = pd.DataFrame(rows).set_index("year")

    long = result.reset_index().melt("year", var_name="Strategy", value_name="€/MWh")
    fig = px.bar(long, x="year", y="€/MWh", color="Strategy", barmode="group",
                 color_discrete_sequence=[BLUE, ORANGE, AQUA, "#eda100"],
                 title="Average energy cost per MWh, by strategy and year")
    st.plotly_chart(style(fig, 420), width="stretch")

    risk = pd.DataFrame({"Average cost (€/MWh)": result.mean(),
                         "Year-to-year volatility (€/MWh)": result.std()})
    fig = px.scatter(risk.reset_index(names="Strategy"), x="Year-to-year volatility (€/MWh)",
                     y="Average cost (€/MWh)", text="Strategy",
                     color_discrete_sequence=[BLUE], title="Cost vs risk: lower-left is better")
    fig.update_traces(marker_size=14, textposition="top center")
    st.plotly_chart(style(fig), width="stretch")
    st.dataframe(result.round(1), width="stretch")
    st.caption("Simplifications: the fixed price is approximated from the previous year's spot "
               "average; network costs, taxes and imbalance costs are excluded.")
