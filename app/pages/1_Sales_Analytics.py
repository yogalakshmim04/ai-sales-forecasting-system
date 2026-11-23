import pandas as pd
import plotly.express as px
import streamlit as st
from utils import setup, load_transactions

setup("Sales Analytics", "📈")
full = load_transactions()

with st.sidebar:
    dates = st.date_input("Date range", (full["Date"].min(), full["Date"].max()))
    countries = st.multiselect("Country", sorted(full["Country"].unique()))
    products = st.multiselect("Product", sorted(full["Description"].unique()))
    freq = st.radio("View", ["Daily", "Weekly", "Monthly"], horizontal=True)

if len(dates) != 2:
    st.info("Select both start and end date.")
    st.stop()

start, end = pd.Timestamp(dates[0]), pd.Timestamp(dates[1])

base = full
if countries:
    base = base[base["Country"].isin(countries)]
if products:
    base = base[base["Description"].isin(products)]

cur = base[(base["Date"] >= start) & (base["Date"] <= end)]
span = (end - start).days + 1
prev = base[(base["Date"] >= start - pd.Timedelta(days=span)) & (base["Date"] < start)]


def aov(x):
    n = x["InvoiceNo"].nunique()
    return x["Revenue"].sum() / n if n else 0


if cur.empty:
    st.warning("No data for this filter.")
    st.stop()

# ---- KPIs with growth vs previous period ----
rev_now, rev_prev = cur["Revenue"].sum(), prev["Revenue"].sum()
growth = f"{(rev_now - rev_prev) / rev_prev * 100:+.1f}% vs previous period" if rev_prev else None

c = st.columns(4)
c[0].metric("Revenue", f"£{rev_now:,.0f}", growth)
c[1].metric("Units Sold", f"{cur['Quantity'].sum():,}")
c[2].metric("Orders", f"{cur['InvoiceNo'].nunique():,}")
c[3].metric("Avg Order Value", f"£{aov(cur):,.2f}")
if not rev_prev:
    st.caption("No data exists for the previous period (start of dataset), so growth % is not shown.")

# ---- Trend ----
rule = {"Daily": "D", "Weekly": "W-SUN", "Monthly": "MS"}[freq]
s = cur.set_index("Date").resample(rule).agg({"Revenue": "sum", "Quantity": "sum"}).reset_index()
st.plotly_chart(px.line(s, x="Date", y="Revenue", title=f"{freq} Revenue"), width="stretch")

a, b = st.columns(2)
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow = cur.groupby(cur["Date"].dt.day_name())["Revenue"].sum().reindex(days).reset_index()
dow.columns = ["Day", "Revenue"]
a.plotly_chart(px.bar(dow, x="Day", y="Revenue", title="Revenue by Day of Week"), width="stretch")
b.plotly_chart(px.line(s, x="Date", y="Quantity", title=f"{freq} Units Sold"), width="stretch")

# ---- Country performance ----
ctry = cur.groupby("Country").agg(Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique")).reset_index()
a, b = st.columns(2)
a.plotly_chart(px.bar(ctry.nlargest(10, "Revenue"), x="Country", y="Revenue", color="Orders",
                      title="Top 10 Countries (colour = orders)"), width="stretch")
top_days = cur.groupby("Date")["Revenue"].sum().nlargest(5).reset_index()
top_days["Date"] = top_days["Date"].dt.date
b.subheader("Top 5 sales days")
b.dataframe(top_days, width="stretch", hide_index=True)

# ---- Download ----
export = cur.groupby(["Date", "Country", "Description"]).agg(
    Quantity=("Quantity", "sum"), Revenue=("Revenue", "sum")).reset_index()
st.download_button("Download filtered sales CSV", export.to_csv(index=False), "filtered_sales.csv")
st.info("The dataset has no Saturday transactions (the business was closed).")