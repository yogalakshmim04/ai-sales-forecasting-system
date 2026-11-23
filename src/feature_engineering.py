# Feature engineering
import pandas as pd

from src.config import (
    DAILY_SALES_PATH, WEEKLY_SALES_PATH, PROCESSED_DIR, TEST_DAYS, TEST_WEEKS,
)

DAILY_FEATURES = [
    "Month", "Quarter", "Day", "DayOfWeek", "WeekOfYear", "IsMonthStart", "IsMonthEnd",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28", "rolling_std_7",
    "diff_1_7",
]

WEEKLY_FEATURES = [
    "Month", "Quarter", "WeekOfYear",
    "lag_1", "lag_2", "lag_4",
    "rolling_mean_4", "rolling_std_4",
    "lag_1_ratio", "lag_2_ratio", "lag_4_ratio",
]

def make_daily_features(df):

    """Add time, lag and rolling features to daily data (target = Revenue).
Every lag/rolling value uses ONLY past data (shift(1)), so there is no data leakage.
Rows with a missing Revenue (future dates) are supported for forecasting.
    """

    df = df.copy().sort_values("Date").reset_index(drop=True)

    d = df["Date"]
    df["Month"] = d.dt.month
    df["Quarter"] = d.dt.quarter
    df["Day"] = d.dt.day
    df["DayOfWeek"] = d.dt.dayofweek
    df["WeekOfYear"] = d.dt.isocalendar().week.astype(int)
    df["IsMonthStart"] = d.dt.is_month_start.astype(int)
    df["IsMonthEnd"] = d.dt.is_month_end.astype(int)

    for lag in (1, 7, 14, 28):
        df[f"lag_{lag}"] = df["Revenue"].shift(lag)

    past = df["Revenue"].shift(1)          # yesterday-varaikum mattum
    df["rolling_mean_7"] = past.rolling(7).mean()
    df["rolling_mean_14"] = past.rolling(14).mean()
    df["rolling_mean_28"] = past.rolling(28).mean()
    df["rolling_std_7"] = past.rolling(7).std()
    df["diff_1_7"] = df["lag_1"] - df["lag_7"]
    return df


def make_weekly_features(df):
    """Add time + lag + rolling features to weekly data (target = Revenue)."""
    df = df.copy().sort_values("Week").reset_index(drop=True)

    w = df["Week"]
    df["Month"] = w.dt.month
    df["Quarter"] = w.dt.quarter
    df["WeekOfYear"] = w.dt.isocalendar().week.astype(int)

    for lag in (1, 2, 4):
        df[f"lag_{lag}"] = df["Revenue"].shift(lag)

    past = df["Revenue"].shift(1)
    df["rolling_mean_4"] = past.rolling(4).mean()
    df["rolling_std_4"] = past.rolling(4).std()
    for lag in (1, 2, 4):
        df[f"lag_{lag}_ratio"] = df[f"lag_{lag}"] / df["rolling_mean_4"]
    return df


def time_split(df, test_size):
    """Time-based split: last `test_size` rows = test. No shuffling."""
    train = df.iloc[:-test_size].copy()
    test = df.iloc[-test_size:].copy()
    return train, test


def get_daily_dataset():
    """Returns train, test for daily forecasting."""
    daily = pd.read_csv(DAILY_SALES_PATH, parse_dates=["Date"])
    data = make_daily_features(daily).dropna(subset=DAILY_FEATURES + ["Revenue"])
    return time_split(data.reset_index(drop=True), TEST_DAYS)


def get_weekly_dataset():
    """Returns train, test for weekly forecasting."""
    weekly = pd.read_csv(WEEKLY_SALES_PATH, parse_dates=["Week"])
    data = make_weekly_features(weekly).dropna(subset=WEEKLY_FEATURES + ["Revenue"])
    return time_split(data.reset_index(drop=True), TEST_WEEKS)


def run_feature_engineering():
    daily_train, daily_test = get_daily_dataset()
    weekly_train, weekly_test = get_weekly_dataset()

    pd.concat([daily_train, daily_test]).to_csv(PROCESSED_DIR / "daily_features.csv", index=False)
    pd.concat([weekly_train, weekly_test]).to_csv(PROCESSED_DIR / "weekly_features.csv", index=False)

    print(f"Daily  -> train: {len(daily_train)}, test: {len(daily_test)}")
    print(f"Weekly -> train: {len(weekly_train)}, test: {len(weekly_test)}")
    print(f"Daily test period : {daily_test['Date'].min().date()} to {daily_test['Date'].max().date()}")
    print(f"Weekly test period: {weekly_test['Week'].min().date()} to {weekly_test['Week'].max().date()}")


if __name__ == "__main__":
    run_feature_engineering()