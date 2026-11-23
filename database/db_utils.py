# Database utilities
import sqlite3
from contextlib import closing

import pandas as pd

from src.config import DB_PATH, CLEAN_DATA_PATH, DAILY_SALES_PATH, WEEKLY_SALES_PATH


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def save_table(df, table_name, if_exists="replace"):
    """Save a DataFrame as a SQLite table."""
    with closing(get_connection()) as conn:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    print(f"Saved table '{table_name}' ({len(df)} rows)")


def load_table(table_name, parse_dates=None):
    """Load a SQLite table as a DataFrame."""
    with closing(get_connection()) as conn:
        return pd.read_sql(f"SELECT * FROM {table_name}", conn, parse_dates=parse_dates)


def list_tables():
    with closing(get_connection()) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    return [r[0] for r in rows]


def build_database():
    """Load processed CSVs into the SQLite database."""
    transactions = pd.read_csv(CLEAN_DATA_PATH, parse_dates=["InvoiceDate", "Date"])
    daily = pd.read_csv(DAILY_SALES_PATH, parse_dates=["Date"])
    weekly = pd.read_csv(WEEKLY_SALES_PATH, parse_dates=["Week"])

    save_table(transactions, "transactions")
    save_table(daily, "daily_sales")
    save_table(weekly, "weekly_sales")
    print("Tables:", list_tables())


if __name__ == "__main__":
    build_database()