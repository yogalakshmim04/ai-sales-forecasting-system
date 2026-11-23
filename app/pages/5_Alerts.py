# Alerts page
import streamlit as st
from utils import setup
from src.alerts import generate_alerts

setup("Alerts & Business Insights", "🚨")
show = {"warning": st.warning, "success": st.success, "info": st.info, "error": st.error}
for a in generate_alerts():
    show[a["Level"]](f"**{a['Title']}**: {a['Message']}")
st.caption("Alerts are generated from calculated metrics and forecasts, not written manually.")