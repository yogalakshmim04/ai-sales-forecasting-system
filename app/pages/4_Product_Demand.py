# Product & Demand page
import plotly.express as px
import streamlit as st
from utils import setup, load_csv
from src.demand_analysis import inventory_check

setup("Product & Demand", "📦")
prod = load_csv("product_summary.csv")
dem = load_csv("product_demand.csv")

by = st.radio("Rank by", ["Revenue", "Quantity"], horizontal=True)
a, b = st.columns(2)
a.plotly_chart(px.bar(prod.nlargest(10, by), x=by, y="Description", orientation="h",
                      title=f"Top 10 by {by}").update_yaxes(autorange="reversed"), use_container_width=True)
b.plotly_chart(px.bar(prod.nsmallest(10, by), x=by, y="Description", orientation="h",
                      title=f"Lowest 10 by {by}").update_yaxes(autorange="reversed"), use_container_width=True)

st.subheader("Predicted demand (next 4 weeks)")
trend = st.multiselect("Trend", ["Increasing", "Stable", "Decreasing"], default=["Increasing", "Stable", "Decreasing"])
d = dem[dem["Trend"].isin(trend)]
st.dataframe(d[["StockCode", "Description", "Previous_4W", "Recent_4W", "Change_%", "Trend", "Predicted_Next_4W"]].head(200),
             use_container_width=True, hide_index=True)

st.subheader("Inventory check")
choice = st.selectbox("Product", dem["Description"].head(200))
row = dem[dem["Description"] == choice].iloc[0]
stock = st.number_input("Current stock (units)", min_value=0, value=1000, step=50)
status = inventory_check(row["Predicted_Next_4W"], stock)
msg = f"Predicted demand: {row['Predicted_Next_4W']:,.0f} units vs stock {stock:,} : **{status}**"
(st.error if status == "Shortage risk" else st.warning if status == "Low buffer" else st.success)(msg)
st.caption("The dataset has no stock column, so the stock value must be entered manually.")