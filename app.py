"""
BI Dashboard (Streamlit)
Run:  streamlit run app.py
"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from forecast import load_data, monthly_series, run_pipeline

st.set_page_config(page_title="AI Sales Forecasting & BI", layout="wide")
st.title("AI Powered Sales Forecasting & Business Intelligence")


@st.cache_data
def get_data():
    return load_data()


@st.cache_data
def get_forecast(series, horizon):
    return run_pipeline(series, horizon=horizon)


df = get_data()

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
regions = st.sidebar.multiselect("Region", sorted(df["region"].unique()),
                                 default=sorted(df["region"].unique()))
cats = st.sidebar.multiselect("Category", sorted(df["category"].unique()),
                              default=sorted(df["category"].unique()))
min_d, max_d = df["date"].min().date(), df["date"].max().date()
date_range = st.sidebar.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
horizon = st.sidebar.slider("Forecast months", 3, 12, 6)

if len(date_range) != 2:
    st.info("Select both start and end date.")
    st.stop()

mask = (
    df["region"].isin(regions)
    & df["category"].isin(cats)
    & (df["date"].dt.date >= date_range[0])
    & (df["date"].dt.date <= date_range[1])
)
fdf = df[mask]
if fdf.empty:
    st.warning("No data for these filters.")
    st.stop()

s = monthly_series(fdf)

# ---------------- KPIs ----------------
total = fdf["sales"].sum()
last, prev = s.iloc[-1], s.iloc[-2] if len(s) > 1 else s.iloc[-1]
growth = (last - prev) / prev * 100 if prev else 0

try:
    out = get_forecast(s, horizon)
except ValueError as e:
    st.warning(f"{e} Widen the date range or filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", f"{total:,.0f}")
c2.metric("Last Month Sales", f"{last:,.0f}", f"{growth:.1f}% MoM")
c3.metric("Next Month Forecast", f"{out['future'].iloc[0]:,.0f}")
c4.metric("Best Model", out["best"])

# ---------------- Forecast chart ----------------
st.subheader("Actual vs Forecast")
fig = go.Figure()
fig.add_trace(go.Scatter(x=s.index, y=s.values, name="Actual", mode="lines"))
fig.add_trace(go.Scatter(x=out["test"].index, y=out["test"]["predicted"],
                         name="Model on test months", mode="lines", line=dict(dash="dot")))
fig.add_trace(go.Scatter(x=out["future"].index, y=out["future"].values,
                         name="Future forecast", mode="lines+markers", line=dict(dash="dash")))
fig.update_layout(height=420, xaxis_title="Month", yaxis_title="Sales")
st.plotly_chart(fig, use_container_width=True)

# ---------------- Top categories / regions ----------------
left, right = st.columns(2)
with left:
    st.subheader("Sales by Category")
    g = fdf.groupby("category", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    st.plotly_chart(px.bar(g, x="category", y="sales"), use_container_width=True)
with right:
    st.subheader("Sales by Region")
    g = fdf.groupby("region", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    st.plotly_chart(px.bar(g, x="region", y="sales"), use_container_width=True)

# ---------------- Model accuracy + download ----------------
st.subheader("Model Comparison (lower error = better)")
st.dataframe(out["metrics"], use_container_width=True, hide_index=True)

st.subheader("Forecast Table")
ft = out["future"].round(0).reset_index()
ft.columns = ["Month", "Forecast"]
st.dataframe(ft, use_container_width=True, hide_index=True)
st.download_button("Download forecast CSV", ft.to_csv(index=False), "forecast.csv", "text/csv")
