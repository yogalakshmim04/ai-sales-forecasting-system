import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils import setup
from src.upload_forecast import (
    MIN_DAYS, aggregate_daily, clean_upload, forecast_upload, guess_column, read_upload,
)

setup("Upload Your Data", "📤")
st.write("Upload your own sales file (CSV or Excel). The system cleans it, trains a model "
         "on it and forecasts future daily revenue.")

sample = pd.DataFrame({"Date": ["2024-01-01", "2024-01-01", "2024-01-02"],
                       "Quantity": [3, 1, 5], "UnitPrice": [10.5, 99.0, 4.25]})
st.download_button("Download sample template", sample.to_csv(index=False), "sales_template.csv")

file = st.file_uploader("Sales file", type=["csv", "xlsx", "xls"])
if file is None:
    st.info("Required: a date column, plus either Quantity and Price columns "
            "or a ready-made Revenue column.")
    st.stop()

try:
    raw = read_upload(file)
except Exception as error:
    st.error(f"The file could not be read: {error}")
    st.stop()

st.subheader("1. Preview")
st.dataframe(raw.head(10), width="stretch", hide_index=True)

st.subheader("2. Map your columns")
columns = list(raw.columns)
left, middle, right = st.columns(3)
date_col = left.selectbox("Date column", columns,
                          index=columns.index(guess_column(columns, ["date", "time"])))
revenue_options = ["(calculate from Quantity x Price)"] + columns
revenue_choice = right.selectbox("Revenue column (optional)", revenue_options)
use_revenue = revenue_choice != revenue_options[0]

qty_col = price_col = None
if not use_revenue:
    qty_col = left.selectbox("Quantity column", columns,
                             index=columns.index(guess_column(columns, ["quantity", "qty", "units"])))
    price_col = middle.selectbox("Price column", columns,
                                 index=columns.index(guess_column(columns, ["price", "rate", "cost"])))
dayfirst = st.checkbox("Dates are day-first (DD/MM/YYYY)")
horizon = st.slider("Forecast horizon (days)", 7, 28, 14)

if not st.button("Clean data and forecast", type="primary"):
    st.stop()

clean, report = clean_upload(raw, date_col, qty_col, price_col,
                             revenue_choice if use_revenue else None, dayfirst)
st.subheader("3. Data quality report")
st.dataframe(pd.DataFrame(report.items(), columns=["Check", "Rows"]),
             width="stretch", hide_index=True)

if clean.empty:
    st.error("No valid rows remain. Check the selected columns.")
    st.stop()

daily = aggregate_daily(clean)
c = st.columns(3)
c[0].metric("Date range", f"{daily['Date'].min():%d %b %Y} to {daily['Date'].max():%d %b %Y}")
c[1].metric("Days covered", f"{len(daily):,}")
c[2].metric("Total revenue", f"{daily['Revenue'].sum():,.0f}")

if len(daily) < 365:
    st.warning("Less than one year of data: seasonal patterns cannot be learned reliably.")

try:
    with st.spinner("Training and validating the model..."):
        forecast, metrics = forecast_upload(daily, horizon)
except ValueError as error:
    st.error(str(error) + f" Upload at least {MIN_DAYS} days of daily history.")
    st.stop()

st.subheader("4. Forecast")
history = daily.tail(90)
fig = go.Figure()
fig.add_scatter(x=history["Date"], y=history["Revenue"], name="Actual", line=dict(color="#1f77b4"))
fig.add_scatter(x=forecast["Date"], y=forecast["Upper"], line=dict(width=0), showlegend=False)
fig.add_scatter(x=forecast["Date"], y=forecast["Lower"], fill="tonexty", line=dict(width=0),
                fillcolor="rgba(255,127,14,0.2)", name="95% range")
fig.add_scatter(x=forecast["Date"], y=forecast["Forecast"], name="Forecast",
                line=dict(color="#ff7f0e", dash="dash"))
fig.update_layout(title="Daily revenue: actual vs forecast", yaxis_title="Revenue")
st.plotly_chart(fig, width="stretch")

k = st.columns(3)
k[0].metric("Forecast total", f"{forecast['Forecast'].sum():,.0f}")
k[1].metric("Forecast daily average", f"{forecast['Forecast'].mean():,.0f}")
k[2].metric("Method used", metrics["Method"])

st.dataframe(pd.DataFrame([metrics]), width="stretch", hide_index=True)
st.download_button("Download forecast CSV", forecast.to_csv(index=False), "uploaded_data_forecast.csv")
st.caption("The model is validated on the last 28 days and compared with a seasonal naive baseline. "
           "The better method is used. Ranges are approximate.")