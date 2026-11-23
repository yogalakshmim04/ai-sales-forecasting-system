import plotly.graph_objects as go
import streamlit as st
from utils import setup, load_transactions, load_csv
from src.product_forecast import product_weekly_series, forecast_product, backtest_product

setup("Product Forecast", "🎯")

df = load_transactions()
summary = load_csv("product_summary.csv").nlargest(300, "Revenue")
summary["Label"] = summary["Description"] + " (" + summary["StockCode"].astype(str) + ")"

label = st.selectbox("Product (top 300 by revenue)", summary["Label"])
code = summary.loc[summary["Label"] == label, "StockCode"].iloc[0]
horizon = st.slider("Weeks ahead", 1, 8, 4)

series = product_weekly_series(df, code)
fc, method = forecast_product(series, horizon)
history = series.tail(26)

fig = go.Figure()
fig.add_scatter(x=history.index, y=history.values, name="Actual units", line=dict(color="#1f77b4"))
fig.add_scatter(x=fc["Week"], y=fc["Upper"], line=dict(width=0), showlegend=False)
fig.add_scatter(x=fc["Week"], y=fc["Lower"], fill="tonexty", line=dict(width=0),
                fillcolor="rgba(255,127,14,0.2)", name="95% range")
fig.add_scatter(x=fc["Week"], y=fc["Forecast"], name="Forecast",
                line=dict(color="#ff7f0e", dash="dash"))
fig.update_layout(title="Weekly units sold: actual vs forecast", yaxis_title="Units")
st.plotly_chart(fig, width="stretch")

c = st.columns(3)
c[0].metric("Forecast total (units)", f"{fc['Forecast'].sum():,.0f}")
c[1].metric("Last 4 weeks (units)", f"{series.iloc[-4:].sum():,.0f}")
c[2].metric("Method", method)

st.subheader("Backtest on the last 6 weeks")
result = backtest_product(series)
st.dataframe([result], width="stretch", hide_index=True)
if result["Model MAE (units)"] < result["Baseline MAE (units)"]:
    st.success("The model beats the 4-week moving average for this product.")
else:
    st.warning("The simple 4-week moving average is as good or better for this product.")

st.download_button("Download product forecast CSV", fc.to_csv(index=False),
                   f"forecast_{code}.csv")
st.caption("Product-level demand is noisy. Products with fewer than 12 active weeks use a "
           "moving average instead of exponential smoothing.")
