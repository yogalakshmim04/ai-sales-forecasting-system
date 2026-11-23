# Customer Analytics page
import plotly.express as px
import streamlit as st
from utils import setup, load_csv

setup("Customer Analytics", "👥")
rfm = load_csv("customer_segments.csv")
summ = load_csv("segment_summary.csv")

c = st.columns(3)
c[0].metric("Identified customers", f"{len(rfm):,}")
c[1].metric("Avg orders / customer", f"{rfm['Frequency'].mean():.1f}")
c[2].metric("Avg spend / customer", f"£{rfm['Monetary'].mean():,.0f}")

st.dataframe(summ, use_container_width=True, hide_index=True)

a, b = st.columns(2)
a.plotly_chart(px.pie(summ, names="Segment", values="Customers", title="Customers by Segment"), use_container_width=True)
b.plotly_chart(px.pie(summ, names="Segment", values="Total_Revenue", title="Revenue by Segment"), use_container_width=True)

st.plotly_chart(px.scatter(rfm, x="Recency", y="Monetary", color="Segment", size="Frequency",
                           log_y=True, hover_data=["CustomerID"], title="RFM Segments"), use_container_width=True)
seg = st.selectbox("Filter segment", ["All"] + sorted(rfm["Segment"].unique()))
view = rfm if seg == "All" else rfm[rfm["Segment"] == seg]
st.subheader("Top customers")
st.dataframe(view.nlargest(20, "Monetary"), use_container_width=True, hide_index=True)
st.caption("Only rows with CustomerID (~75%) are used here.")