import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, ROOT.as_posix())

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from src.config import PROCESSED_DIR, CLEAN_DATA_PATH

# Change to "dark" to switch back to option 1 (also swap .streamlit/config.toml).
THEME = "fresh"

PALETTES = {
    "fresh": ["#6366F1", "#F59E0B", "#10B981", "#EF4444", "#06B6D4", "#8B5CF6"],
    "dark": ["#818CF8", "#FBBF24", "#34D399", "#F87171", "#22D3EE", "#C084FC"],
}
GRID = {"fresh": "rgba(100,116,139,0.18)", "dark": "rgba(148,163,184,0.15)"}
PALETTE = PALETTES[THEME]

# One chart style for every page. The legend sits below the chart so it never overlaps the title.
pio.templates["app"] = go.layout.Template(layout=dict(
    colorway=PALETTE,
    font=dict(family="Inter, Segoe UI, sans-serif", size=13),
    title=dict(font=dict(size=17), y=0.97, yanchor="top"),
    margin=dict(l=10, r=10, t=60, b=70),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(gridcolor=GRID[THEME], zeroline=False),
    legend=dict(orientation="h", yanchor="top", y=-0.12, x=0),
    hoverlabel=dict(font_size=13),
))
pio.templates.default = "app"

CSS_FRESH = """<style>
.block-container {padding-top: 1.5rem; max-width: 1250px;}

/* Page banner */
.hero {
    background: linear-gradient(120deg, #6366F1 0%, #8B5CF6 55%, #06B6D4 100%);
    border-radius: 20px; padding: 26px 32px; margin-bottom: 1.4rem;
    box-shadow: 0 10px 25px rgba(99,102,241,0.30);
}
.hero-title {color: #FFFFFF; font-size: 2rem; font-weight: 800;}
.hero-sub {color: rgba(255,255,255,0.88); font-size: 1.02rem; margin-top: 0.35rem;}

/* Sidebar */
[data-testid="stSidebar"] {background: #FFFFFF; border-right: 1px solid #E0E7FF;}

/* KPI cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E0E7FF;
    border-left: 6px solid #6366F1;
    border-radius: 16px; padding: 16px 20px;
    box-shadow: 0 4px 14px rgba(99,102,241,0.12);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stMetric"]:hover {transform: translateY(-3px); box-shadow: 0 8px 20px rgba(99,102,241,0.22);}
[data-testid="stMetricLabel"] {color: #6B7280;}
[data-testid="stMetricValue"] {font-size: 1.6rem; font-weight: 800; color: #312E81;}

/* Section headings */
.section-title {
    font-size: 1.25rem; font-weight: 800; color: #312E81;
    margin: 1.8rem 0 0.1rem 0; padding-left: 0.7rem; border-left: 5px solid #F59E0B;
}
.section-text {color: #6B7280; margin: 0.2rem 0 0.8rem 0.9rem;}

/* Charts, tables, help boxes */
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
    background: #FFFFFF; border: 1px solid #E0E7FF; border-radius: 16px;
    padding: 8px; box-shadow: 0 4px 14px rgba(99,102,241,0.08);
}
[data-testid="stExpander"] {border: 1px solid #C7D2FE; border-radius: 12px; background: #EEF2FF;}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    border-radius: 10px; border: 1px solid #6366F1; font-weight: 600;
}
#MainMenu, footer {visibility: hidden;}
</style>"""

CSS_DARK = """<style>
.block-container {padding-top: 1.5rem; max-width: 1250px;}
.hero {
    background: linear-gradient(120deg, #312E81 0%, #4F46E5 60%, #0E7490 100%);
    border-radius: 20px; padding: 26px 32px; margin-bottom: 1.4rem;
}
.hero-title {color: #FFFFFF; font-size: 2rem; font-weight: 800;}
.hero-sub {color: rgba(255,255,255,0.85); font-size: 1.02rem; margin-top: 0.35rem;}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #131C2E 0%, #0B1220 100%);
    border-right: 1px solid rgba(129,140,248,0.25);
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(129,140,248,0.22), rgba(34,211,238,0.08));
    border: 1px solid rgba(129,140,248,0.40); border-radius: 16px; padding: 16px 20px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25); transition: transform 0.15s ease;
}
[data-testid="stMetric"]:hover {transform: translateY(-3px);}
[data-testid="stMetricLabel"] {color: #A5B4FC;}
[data-testid="stMetricValue"] {font-size: 1.6rem; font-weight: 700; color: #F8FAFC;}
.section-title {
    font-size: 1.25rem; font-weight: 700; color: #C7D2FE;
    margin: 1.8rem 0 0.1rem 0; padding-left: 0.7rem; border-left: 4px solid #FBBF24;
}
.section-text {color: #94A3B8; margin: 0.2rem 0 0.8rem 0.9rem;}
[data-testid="stPlotlyChart"] {
    background: rgba(19,28,46,0.65); border: 1px solid rgba(148,163,184,0.15);
    border-radius: 16px; padding: 8px;
}
[data-testid="stExpander"] {
    border: 1px solid rgba(129,140,248,0.30); border-radius: 12px; background: rgba(19,28,46,0.5);
}
.stButton > button, .stDownloadButton > button {
    border-radius: 10px; border: 1px solid rgba(129,140,248,0.6);
}
#MainMenu, footer {visibility: hidden;}
</style>"""

CSS = CSS_FRESH if THEME == "fresh" else CSS_DARK


def setup(title, icon="📊", subtitle=""):
    """Page configuration, global style and a gradient banner with title and description."""
    st.set_page_config(page_title=f"{title} | AI Sales Forecasting", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### 🛒 AI Sales Forecasting")
        st.caption("UCI Online Retail dataset\nDec 2010 - Dec 2011")
        st.divider()
    sub = f'<div class="hero-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="hero"><div class="hero-title">{icon} {title}</div>{sub}</div>',
                unsafe_allow_html=True)


def section(title, text=""):
    """Section heading with a short description of what the section answers."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if text:
        st.markdown(f'<div class="section-text">{text}</div>', unsafe_allow_html=True)


def explain(shows, data, how):
    """Collapsible help box placed under a chart."""
    with st.expander("ℹ️ How to read this chart"):
        st.markdown(f"**What it shows:** {shows}")
        st.markdown(f"**Data used:** {data}")
        st.markdown(f"**How to read it:** {how}")


@st.cache_data
def load_transactions():
    return pd.read_csv(
        CLEAN_DATA_PATH, parse_dates=["InvoiceDate", "Date"],
        usecols=["InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice",
                 "CustomerID", "Country", "InvoiceDate", "Revenue", "Date"])


@st.cache_data
def load_csv(name, date_cols=None):
    return pd.read_csv(PROCESSED_DIR / name, parse_dates=date_cols or [])