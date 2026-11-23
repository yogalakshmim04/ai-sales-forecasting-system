# Business alerts
import pandas as pd

from src.config import WEEKLY_SALES_PATH, PROCESSED_DIR
from src.forecasting import forecast_future


def _alert(level, title, message):
    return {"Level": level, "Title": title, "Message": message}


def generate_alerts():
    alerts = []

    # 1. Last full week vs previous 4 weeks
    weekly = pd.read_csv(WEEKLY_SALES_PATH, parse_dates=["Week"])
    last = weekly["Revenue"].iloc[-1]
    base = weekly["Revenue"].iloc[-5:-1].mean()
    change = (last - base) / base * 100
    if change < -10:
        alerts.append(_alert("warning", "Sales declining",
                             f"Last week revenue £{last:,.0f} is {abs(change):.1f}% below the previous 4-week average."))
    elif change > 10:
        alerts.append(_alert("success", "Sales increasing",
                             f"Last week revenue £{last:,.0f} is {change:.1f}% above the previous 4-week average."))
    else:
        alerts.append(_alert("info", "Sales stable",
                             f"Last week revenue is within {abs(change):.1f}% of the 4-week average."))

    # 2. Forecast direction
    fc = forecast_future("weekly", 4)["Forecast"].mean()
    recent = weekly["Revenue"].iloc[-4:].mean()
    fchange = (fc - recent) / recent * 100
    level = "success" if fchange > 5 else "warning" if fchange < -5 else "info"
    alerts.append(_alert(level, "Forecast outlook",
                         f"Next 4 weeks average forecast is {fchange:+.1f}% vs the last 4 weeks."))

    # 3. Product demand
    demand_path = PROCESSED_DIR / "product_demand.csv"
    if demand_path.exists():
        d = pd.read_csv(demand_path)
        inc, dec = (d["Trend"] == "Increasing").sum(), (d["Trend"] == "Decreasing").sum()
        alerts.append(_alert("info", "Product demand trends",
                             f"{inc} products show increasing demand and {dec} show decreasing demand (recent 4 weeks vs previous 4)."))
        top = d.iloc[0]
        alerts.append(_alert("success", "Highest predicted demand",
                             f"{top['Description']} ({top['StockCode']}): about {top['Predicted_Next_4W']:,.0f} units in the next 4 weeks."))

    # 4. Customer concentration
    seg_path = PROCESSED_DIR / "segment_summary.csv"
    if seg_path.exists():
        s = pd.read_csv(seg_path).iloc[0]
        alerts.append(_alert("info", "Customer concentration",
                             f"'{s['Segment']}' is {s['Customers']:.0f} customers but {s['Revenue_Share_%']}% of identified-customer revenue."))
    return alerts


if __name__ == "__main__":
    for a in generate_alerts():
        print(f"[{a['Level'].upper()}] {a['Title']}: {a['Message']}")