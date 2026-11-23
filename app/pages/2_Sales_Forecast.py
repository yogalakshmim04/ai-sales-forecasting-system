# Sales Forecast page
import plotly.graph_objects as go
import streamlit as st
from utils import setup, load_csv
from src.forecasting import forecast_future
import plotly.express as px
from src.report import build_pdf_report

setup("AI Sales Forecast", "🔮")

level = st.radio("Forecast level", ["Daily", "Weekly"], horizontal=True).lower()
col = "Date" if level == "daily" else "Week"
horizon = st.slider("Days ahead" if level == "daily" else "Weeks ahead",
                    1, 28 if level == "daily" else 8, 7 if level == "daily" else 4)

hist = load_csv(f"{level}_sales.csv", [col])
hist = hist.tail(90 if level == "daily" else 26)
fc = forecast_future(level, horizon)

fig = go.Figure()
fig.add_scatter(x=hist[col], y=hist["Revenue"], name="Historical", line=dict(color="#1f77b4"))
fig.add_scatter(x=fc[col], y=fc["Upper"], line=dict(width=0), showlegend=False)
fig.add_scatter(x=fc[col], y=fc["Lower"], fill="tonexty", line=dict(width=0),
                fillcolor="rgba(255,127,14,0.2)", name="95% range")
fig.add_scatter(x=fc[col], y=fc["Forecast"], name="Forecast", line=dict(color="#ff7f0e", dash="dash"))
fig.update_layout(title=f"{level.title()} Revenue Forecast", yaxis_title="Revenue (£)")
st.plotly_chart(fig, use_container_width=True)

c = st.columns(3)
c[0].metric("Forecast total", f"£{fc['Forecast'].sum():,.0f}")
c[1].metric("Forecast average", f"£{fc['Forecast'].mean():,.0f}")
c[2].metric("Peak", f"£{fc['Forecast'].max():,.0f}")
st.download_button("Download forecast CSV", fc.to_csv(index=False), f"{level}_forecast.csv")

st.subheader("Model comparison (test set)")
comp = load_csv(f"{level}_model_comparison.csv")
st.dataframe(comp, use_container_width=True, hide_index=True)

pred = load_csv(f"{level}_test_predictions.csv", [col])
models = st.multiselect("Actual vs Predicted", [c for c in pred.columns if c not in (col, "Actual")],
                        default=["Random Forest"])
fig2 = go.Figure()
fig2.add_scatter(x=pred[col], y=pred["Actual"], name="Actual", line=dict(color="black"))
for m in models:
    fig2.add_scatter(x=pred[col], y=pred[m], name=m)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Feature importance")
importance = load_csv(f"{level}_feature_importance.csv")
model_options = list(importance["Model"].unique())
default_index = model_options.index("Random Forest") if "Random Forest" in model_options else 0
chosen = st.selectbox("Model", model_options, index=default_index)

chosen_importance = importance[importance["Model"] == chosen].sort_values("Importance")
st.plotly_chart(
    px.bar(chosen_importance, x="Importance", y="Feature", orientation="h",
           title=f"{chosen}: which features drive the forecast"),
    use_container_width=True)
st.caption("Tree models show each feature's share of total error reduction. "
           "Linear Regression shows the absolute standardized coefficient.")

st.subheader("Business report")


@st.cache_data(ttl=600)
def make_report():
    return build_pdf_report()


st.download_button("Download PDF report", make_report(), "sales_forecast_report.pdf",
                   mime="application/pdf")

st.warning("Limitations: the dataset covers only one year. The model does not know about holiday closures "
           "(for example the Christmas break). At weekly level a simple baseline beats the ML models, "
           "while at daily level the ML models are better.")