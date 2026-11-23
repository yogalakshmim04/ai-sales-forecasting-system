# Data preprocessing
import pandas as pd

from src.config import (
    RAW_DATA_PATH, PROCESSED_DIR, CLEAN_DATA_PATH,
    DAILY_SALES_PATH, WEEKLY_SALES_PATH,
)


def load_raw_data():
    """Load the raw UCI Online Retail Excel file."""
    print("Loading raw data (reading the Excel file may take a minute)...")
    df = pd.read_excel(RAW_DATA_PATH, dtype={"InvoiceNo": str, "StockCode": str})
    print(f"Raw shape: {df.shape}")
    return df


def clean_data(df):
    """Clean raw transactions and add Revenue column."""
    n_start = len(df)

    # 1. Remove exact duplicate rows
    df = df.drop_duplicates()

    # 2. Convert date column
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # 3. Remove cancellations (InvoiceNo starts with 'C')
    n_cancel = df["InvoiceNo"].str.startswith("C").sum()
    df = df[~df["InvoiceNo"].str.startswith("C")]

    # 4. Remove invalid quantity and price
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # 5. Keep only real products (StockCode starts with 5 digits).
    #    Removes non-product codes like POST, D, M, BANK CHARGES, DOT, etc.
    df = df[df["StockCode"].str.match(r"^\d{5}", na=False)]

    # 6. Clean product description
    df["Description"] = df["Description"].astype(str).str.strip().str.upper()

    # 7. Add Revenue and Date columns
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["Date"] = df["InvoiceDate"].dt.normalize()

# 8. CustomerID: keep missing values (needed for sales analytics);
#    they are dropped only for customer analytics
    df["CustomerID"] = df["CustomerID"].astype("Int64")

    df = df.reset_index(drop=True)

    print(f"Cancelled rows removed : {n_cancel}")
    print(f"Rows: {n_start} -> {len(df)}")
    print(f"Missing CustomerID     : {df['CustomerID'].isna().sum()}")
    return df


def get_customer_data(df):
    """Rows that have a CustomerID (used for RFM and K-Means)."""
    return df.dropna(subset=["CustomerID"]).copy()


def aggregate_daily(df):
    """Daily sales. Days with no transactions are filled with 0."""
    daily = df.groupby("Date").agg(
        Revenue=("Revenue", "sum"),
        Quantity=("Quantity", "sum"),
        Orders=("InvoiceNo", "nunique"),
        Customers=("CustomerID", "nunique"),
    )
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0)
    daily.index.name = "Date"
    return daily.reset_index()


def aggregate_weekly(df):
    """Weekly sales (week ends on Sunday). First and last weeks are partial, so dropped."""
    weekly = df.groupby(pd.Grouper(key="Date", freq="W-SUN")).agg(
        Revenue=("Revenue", "sum"),
        Quantity=("Quantity", "sum"),
        Orders=("InvoiceNo", "nunique"),
        Customers=("CustomerID", "nunique"),
    )
    weekly = weekly.iloc[1:-1]
    weekly.index.name = "Week"
    return weekly.reset_index()


def run_preprocessing():
    """Full pipeline: raw -> clean -> daily/weekly -> save CSVs."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = clean_data(load_raw_data())
    daily = aggregate_daily(df)
    weekly = aggregate_weekly(df)

    df.to_csv(CLEAN_DATA_PATH, index=False)
    daily.to_csv(DAILY_SALES_PATH, index=False)
    weekly.to_csv(WEEKLY_SALES_PATH, index=False)

    print(f"\nDaily rows : {len(daily)}")
    print(f"Weekly rows: {len(weekly)}")
    print(f"Total revenue: £{df['Revenue'].sum():,.2f}")
    print("Saved to data/processed/")
    return df, daily, weekly


if __name__ == "__main__":
    run_preprocessing()