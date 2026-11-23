# Product analysis
import pandas as pd

from src.config import CLEAN_DATA_PATH, PROCESSED_DIR


def load_transactions():
    return pd.read_csv(CLEAN_DATA_PATH, parse_dates=["InvoiceDate", "Date"])


def product_summary(df):
    """One row per product: quantity, revenue, orders, customers."""
    names = df.groupby("StockCode")["Description"].agg(lambda x: x.mode().iat[0])
    summary = df.groupby("StockCode").agg(
        Quantity=("Quantity", "sum"),
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
        Customers=("CustomerID", "nunique"),
    ).join(names.rename("Description"))
    return summary.reset_index()


def top_products(summary, n=10, by="Revenue"):
    return summary.nlargest(n, by)


def low_products(summary, n=10, by="Revenue"):
    return summary.nsmallest(n, by)


def country_summary(df):
    c = df.groupby("Country").agg(Revenue=("Revenue", "sum"),
                                  Orders=("InvoiceNo", "nunique")).reset_index()
    return c.sort_values("Revenue", ascending=False)


def run_product_analysis():
    df = load_transactions()
    summary = product_summary(df)
    summary.to_csv(PROCESSED_DIR / "product_summary.csv", index=False)
    country_summary(df).to_csv(PROCESSED_DIR / "country_summary.csv", index=False)

    print(f"Products: {len(summary)}")
    print("\n=== TOP 5 BY REVENUE ===")
    print(top_products(summary, 5)[["StockCode", "Description", "Quantity", "Revenue"]].to_string(index=False))
    print("\n=== TOP 5 BY QUANTITY ===")
    print(top_products(summary, 5, "Quantity")[["StockCode", "Description", "Quantity", "Revenue"]].to_string(index=False))
    return summary


if __name__ == "__main__":
    run_product_analysis()
    