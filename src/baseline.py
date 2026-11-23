# Baseline models
from src.feature_engineering import get_daily_dataset, get_weekly_dataset
from src.evaluation import compare_models


def daily_baselines(test):
    y = test["Revenue"]
    return {
        "Naive (yesterday)": (y, test["lag_1"]),
        "Seasonal Naive (same day last week)": (y, test["lag_7"]),
        "Moving Average (7-day)": (y, test["rolling_mean_7"]),
    }


def weekly_baselines(test):
    y = test["Revenue"]
    return {
        "Naive (last week)": (y, test["lag_1"]),
        "Moving Average (4-week)": (y, test["rolling_mean_4"]),
    }


def run_baselines():
    _, daily_test = get_daily_dataset()
    _, weekly_test = get_weekly_dataset()

    print("\n=== DAILY BASELINES ===")
    print(compare_models(daily_baselines(daily_test)).to_string(index=False))

    print("\n=== WEEKLY BASELINES ===")
    print(compare_models(weekly_baselines(weekly_test)).to_string(index=False))


if __name__ == "__main__":
    run_baselines()