import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils import setup, section, explain, load_transactions, load_csv
from src.forecasting import forecast_future, get_method

setup("AI E-Commerce Analytics", "🛒",
      "A one-page summary: how the business is doing now and what is expected next.")

df = load_transactions()
weekly = load_csv("weekly_sales.csv", ["Week"])


@st.cache_data
def get_forecast():
    return forecast_future("weekly", 4)


fc = get_forecast()

revenue = df["Revenue"].sum()
orders = df["InvoiceNo"].nunique()
last4 = weekly["Revenue"].iloc[-4:].sum()
prev4 = weekly["Revenue"].iloc[-8:-4].sum()
growth = (last4 - prev4) / prev4 * 100
top_product = df.groupby("Description")["Revenue"].sum().idxmax()

# ---------------- Business now ----------------
section("📌 How is the business doing?", "Totals for the whole dataset period.")
c = st.columns(5)
c[0].metric("Total Revenue", f"£{revenue:,.0f}")
c[1].metric("Orders", f"{orders:,}")
c[2].metric("Customers", f"{df['CustomerID'].nunique():,}")
c[3].metric("Products", f"{df['StockCode'].nunique():,}")
c[4].metric("Avg Order Value", f"£{revenue / orders:,.2f}")

# ---------------- What next ----------------
section("🔮 What is expected next?", "Forecast for the next 4 weeks and the latest growth trend.")
d = st.columns(3)
d[0].metric("Forecast: next 4 weeks", f"£{fc['Forecast'].sum():,.0f}")
d[1].metric("Growth (last 4 vs previous 4 weeks)", f"{growth:+.1f}%", f"{growth:+.1f}%")
d[2].metric("Top product by revenue", top_product.title())

hist = weekly.tail(26)
fig = go.Figure()
fig.add_scatter(x=hist["Week"], y=hist["Revenue"], name="Actual weekly revenue")
fig.add_scatter(x=fc["Week"], y=fc["Upper"], line=dict(width=0), showlegend=False)
fig.add_scatter(x=fc["Week"], y=fc["Lower"], fill="tonexty", line=dict(width=0),
                fillcolor="rgba(245,158,11,0.2)", name="95% range")
fig.add_scatter(x=fc["Week"], y=fc["Forecast"], name="Forecast",
                line=dict(color="#F59E0B", dash="dash"))
fig.update_layout(title="Weekly revenue: actual vs forecast", yaxis_title="Revenue (£)")
st.plotly_chart(fig, width="stretch")
explain("Revenue of the last 26 weeks (solid line) and the predicted revenue for the next 4 weeks (dashed line).",
        "weekly_sales.csv, built from all cleaned transactions. Forecast method: "
        f"{get_method('weekly')}.",
        "The shaded band is the likely range. A wide band means the forecast is uncertain. "
        "The model does not know about holiday closures, so late-December values may be too high.")

# ---------------- Trends ----------------
section("📅 Monthly trends", "How revenue and the number of orders changed month by month.")
month = df["Date"].dt.to_period("M").dt.to_timestamp()
m = df.groupby(month).agg(Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique")).reset_index()
a, b = st.columns(2)
a.plotly_chart(px.bar(m, x="Date", y="Revenue", title="Monthly revenue"), width="stretch")
b.plotly_chart(px.line(m, x="Date", y="Orders", markers=True, title="Monthly orders"), width="stretch")
explain("Total revenue (bars) and unique invoices (line) for each month.",
        "Cleaned transactions: cancellations, invalid prices and non-product codes removed.",
        "Rising bars from September to November show the pre-Christmas peak. "
        "December 2011 is low because the data ends on 9 December.")

# ---------------- Top lists ----------------
section("🏆 What sells and where?", "The 10 products and 10 countries that bring in the most revenue.")
a, b = st.columns(2)
top = df.groupby("Description")["Revenue"].sum().nlargest(10).reset_index()
a.plotly_chart(px.bar(top, x="Revenue", y="Description", orientation="h",
                      title="Top 10 products").update_yaxes(autorange="reversed"), width="stretch")
ctry = df.groupby("Country")["Revenue"].sum().nlargest(10).reset_index()
b.plotly_chart(px.bar(ctry, x="Revenue", y="Country", orientation="h",
                      title="Top 10 countries").update_yaxes(autorange="reversed"), width="stretch")
explain("Revenue ranking of products and countries.",
        "Revenue = Quantity x UnitPrice, summed per product description and per country.",
        "Longer bar = more revenue. The United Kingdom dominates because the retailer is UK-based.")

with st.expander("🧭 Guide to the other pages"):
    st.markdown("""
- **Sales Analytics**: filter by date, country and product; see growth and day-of-week patterns.
- **Sales Forecast**: daily and weekly predictions, model comparison and feature importance.
- **Customer Analytics**: customer groups (RFM + K-Means).
- **Product Demand**: top and low products, 4-week demand outlook, stock check.
- **Alerts**: automatic business insights.
- **Product Forecast**: weekly forecast for a single product.
- **Upload Forecast**: forecast your own sales file.
""")
st.caption("Data: UCI Online Retail (Dec 2010 - Dec 2011). First and last periods are partial.")