# Demand analysis
import numpy as np
import pandas as pd

from src.config import CLEAN_DATA_PATH, WEEKLY_SALES_PATH, PROCESSED_DIR
from src.forecasting import forecast_future

DEMAND_PATH = PROCESSED_DIR / "product_demand.csv"


def weekly_product_qty(df, last_week_end):
    """Quantity per product per full week."""
    df = df[df["Date"] <= last_week_end]
    return (df.groupby(["StockCode", pd.Grouper(key="Date", freq="W-SUN")])["Quantity"]
              .sum().unstack(fill_value=0))


def demand_ratio():
    """Forecasted weekly revenue vs recent actual (limited to 0.5 - 1.5)."""
    weekly = pd.read_csv(WEEKLY_SALES_PATH, parse_dates=["Week"])
    recent_avg = weekly["Revenue"].iloc[-4:].mean()
    forecast_avg = forecast_future("weekly", 4)["Forecast"].mean()
    return float(np.clip(forecast_avg / recent_avg, 0.5, 1.5))


def build_demand_table():
    df = pd.read_csv(CLEAN_DATA_PATH, parse_dates=["InvoiceDate", "Date"])
    weekly = pd.read_csv(WEEKLY_SALES_PATH, parse_dates=["Week"])
    w = weekly_product_qty(df, weekly["Week"].max())

    recent = w.iloc[:, -4:].sum(axis=1)
    previous = w.iloc[:, -8:-4].sum(axis=1)

    t = pd.DataFrame({"Previous_4W": previous, "Recent_4W": recent})
    t = t[(t["Previous_4W"] > 0) & (t["Previous_4W"] + t["Recent_4W"] >= 100)]  # noise filter
    t["Change_%"] = ((t["Recent_4W"] - t["Previous_4W"]) / t["Previous_4W"] * 100).round(1)
    t["Trend"] = np.where(t["Change_%"] > 20, "Increasing",
                  np.where(t["Change_%"] < -20, "Decreasing", "Stable"))

    ratio = demand_ratio()
    t["Predicted_Next_4W"] = (t["Recent_4W"] * ratio).round(0)

    names = df.groupby("StockCode")["Description"].agg(lambda x: x.mode().iat[0])
    t = t.join(names.rename("Description")).reset_index()
    return t.sort_values("Predicted_Next_4W", ascending=False), ratio


def inventory_check(predicted_demand, current_stock, buffer=1.2):
    """Stock input user kudukkanum (dataset la stock column illa)."""
    if current_stock < predicted_demand:
        return "Shortage risk"
    if current_stock < predicted_demand * buffer:
        return "Low buffer"
    return "Sufficient"


def run_demand_analysis():
    table, ratio = build_demand_table()
    table.to_csv(DEMAND_PATH, index=False)
    print(f"Demand ratio (forecast / recent): {ratio:.2f}")
    print(f"Products analysed: {len(table)}")
    print(table["Trend"].value_counts().to_string())
    print("\n=== TOP 5 PREDICTED DEMAND ===")
    print(table.head(5)[["StockCode", "Description", "Recent_4W", "Predicted_Next_4W", "Trend"]].to_string(index=False))
    return table


if __name__ == "__main__":
    run_demand_analysis()