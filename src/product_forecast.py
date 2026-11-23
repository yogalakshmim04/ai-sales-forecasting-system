import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.config import WEEKLY_SALES_PATH

MIN_ACTIVE_WEEKS = 12   # fewer active weeks than this -> use a simple moving average


def get_week_index():
    """Complete weeks used in the weekly sales table (partial first/last weeks excluded)."""
    weekly = pd.read_csv(WEEKLY_SALES_PATH, parse_dates=["Week"])
    return pd.DatetimeIndex(weekly["Week"])


def product_weekly_series(df, stock_code):
    """Weekly units sold for one product. Weeks without sales are filled with 0."""
    weeks = get_week_index()
    sub = df[df["StockCode"] == stock_code]
    series = sub.groupby(pd.Grouper(key="Date", freq="W-SUN"))["Quantity"].sum()
    return series.reindex(weeks, fill_value=0).astype(float)


def _fit_forecast(series, horizon):
    """Return (predictions, in-sample residuals, method name)."""
    if (series > 0).sum() >= MIN_ACTIVE_WEEKS:
        try:
            model = ExponentialSmoothing(
                series.values, trend="add", damped_trend=True,
                initialization_method="estimated").fit()
            pred = np.clip(model.forecast(horizon), 0, None)
            residuals = series.values - model.fittedvalues
            return pred, residuals, "Damped Holt smoothing"
        except Exception:
            pass  # fall back to the moving average below

    level = series.iloc[-4:].mean()
    pred = np.full(horizon, level)
    rolling = series.rolling(4).mean().shift(1)
    residuals = (series - rolling).values[4:]
    return pred, residuals, "4-week moving average"


def forecast_product(series, horizon):
    """Forecast weekly units for one product, with a 95% range."""
    pred, residuals, method = _fit_forecast(series, horizon)
    margin = 1.96 * float(np.nanstd(residuals))
    dates = [series.index[-1] + pd.Timedelta(days=7 * (i + 1)) for i in range(horizon)]
    fc = pd.DataFrame({
        "Week": dates,
        "Forecast": pred,
        "Lower": np.clip(pred - margin, 0, None),
        "Upper": pred + margin,
    })
    return fc, method


def backtest_product(series, test_weeks=6):
    """Compare the model with a 4-week moving average on the last `test_weeks` weeks."""
    train, test = series.iloc[:-test_weeks], series.iloc[-test_weeks:]
    pred, _, method = _fit_forecast(train, test_weeks)
    baseline = np.full(test_weeks, train.iloc[-4:].mean())
    return {
        "Method": method,
        "Model MAE (units)": round(float(np.abs(test.values - pred).mean()), 1),
        "Baseline MAE (units)": round(float(np.abs(test.values - baseline).mean()), 1),
    }