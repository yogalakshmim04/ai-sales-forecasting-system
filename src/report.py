import io
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.alerts import generate_alerts
from src.config import PROCESSED_DIR, WEEKLY_SALES_PATH
from src.forecasting import forecast_future

styles = getSampleStyleSheet()


def _table(df):
    data = [list(df.columns)] + df.astype(str).values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5fa")]),
    ]))
    return table


def _forecast_chart(history, forecast):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(history["Week"], history["Revenue"], label="Actual", color="#1f77b4")
    ax.fill_between(forecast["Week"], forecast["Lower"], forecast["Upper"],
                    color="#ff7f0e", alpha=0.2, label="95% range")
    ax.plot(forecast["Week"], forecast["Forecast"], "--", color="#ff7f0e", label="Forecast")
    ax.set_ylabel("Revenue (GBP)")
    ax.legend()
    fig.autofmt_xdate()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return Image(buffer, width=16 * cm, height=6.5 * cm)


def build_pdf_report(weeks=4):
    """Build the business report and return it as PDF bytes."""
    weekly = pd.read_csv(WEEKLY_SALES_PATH, parse_dates=["Week"])
    forecast = forecast_future("weekly", weeks)
    countries = pd.read_csv(PROCESSED_DIR / "country_summary.csv")
    products = pd.read_csv(PROCESSED_DIR / "product_summary.csv")
    comparison = pd.read_csv(PROCESSED_DIR / "weekly_model_comparison.csv")
    segments = pd.read_csv(PROCESSED_DIR / "segment_summary.csv")

    story = [
        Paragraph("AI Sales Forecasting and E-Commerce Analytics Report", styles["Title"]),
        Paragraph(f"Generated on {date.today():%d %B %Y}. Data: UCI Online Retail "
                  "(Dec 2010 - Dec 2011).", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("1. Business summary", styles["Heading2"]),
        Paragraph(f"Total revenue: GBP {countries['Revenue'].sum():,.0f}. "
                  f"Last full week: GBP {weekly['Revenue'].iloc[-1]:,.0f}. "
                  f"Forecast for the next {weeks} weeks: GBP {forecast['Forecast'].sum():,.0f}.",
                  styles["Normal"]),
        Spacer(1, 8),
        _forecast_chart(weekly.tail(26), forecast),
        Paragraph("2. Weekly forecast", styles["Heading2"]),
        _table(forecast.assign(Week=forecast["Week"].dt.date).round(0)),
        Paragraph("3. Model comparison (weekly, test set)", styles["Heading2"]),
        _table(comparison),
        Paragraph("4. Top 10 products by revenue", styles["Heading2"]),
        _table(products.nlargest(10, "Revenue")[["StockCode", "Description", "Quantity", "Revenue"]].round(0)),
        Paragraph("5. Top 5 countries by revenue", styles["Heading2"]),
        _table(countries.head(5).round(0)),
        Paragraph("6. Customer segments", styles["Heading2"]),
        _table(segments),
        Paragraph("7. Alerts and insights", styles["Heading2"]),
    ]
    for alert in generate_alerts():
        story.append(Paragraph(f"<b>{alert['Title']}</b>: {alert['Message']}", styles["Normal"]))
        story.append(Spacer(1, 4))

    story += [
        Paragraph("8. Limitations", styles["Heading2"]),
        Paragraph("The dataset covers one year only, so yearly seasonality cannot be learned. "
                  "The model does not know about holiday closures. Forecast ranges are "
                  "approximate and widen in reality for longer horizons.", styles["Normal"]),
    ]

    buffer = io.BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=2 * cm, bottomMargin=2 * cm).build(story)
    return buffer.getvalue()