import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config import RANDOM_STATE
from src.feature_engineering import DAILY_FEATURES, make_daily_features

MIN_DAYS = 120      # minimum days of history required
TEST_DAYS = 28      # last 28 days are used to validate the model


def read_upload(file):
    """Read an uploaded CSV or Excel file into a DataFrame."""
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="ISO-8859-1")


def guess_column(columns, keywords):
    """Return the first column whose name contains one of the keywords."""
    for column in columns:
        if any(k in str(column).lower() for k in keywords):
            return column
    return columns[0]


def clean_upload(df, date_col, qty_col, price_col, revenue_col=None, dayfirst=False):
    """Validate and clean uploaded sales data.

    Returns a DataFrame with columns Date and Revenue, and a dictionary
    describing how many rows were removed and why.
    """
    report = {"Rows uploaded": len(df)}
    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=dayfirst)

    if revenue_col:
        out["Revenue"] = pd.to_numeric(df[revenue_col], errors="coerce")
    else:
        quantity = pd.to_numeric(df[qty_col], errors="coerce")
        price = pd.to_numeric(df[price_col], errors="coerce")
        out["Revenue"] = quantity * price

    invalid = out["Date"].isna() | out["Revenue"].isna()
    report["Removed: invalid date or number"] = int(invalid.sum())
    out = out[~invalid]

    before = len(out)
    out = out.drop_duplicates()
    report["Removed: duplicate rows"] = before - len(out)

    before = len(out)
    out = out[out["Revenue"] > 0]
    report["Removed: zero or negative revenue"] = before - len(out)

    out["Date"] = out["Date"].dt.normalize()
    report["Rows used"] = len(out)
    return out.reset_index(drop=True), report


def aggregate_daily(clean):
    """Daily revenue. Days without sales are filled with 0."""
    daily = clean.groupby("Date")["Revenue"].sum()
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0.0)
    daily.index.name = "Date"
    return daily.reset_index()


def _new_model():
    return RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)


def forecast_upload(daily, horizon):
    """Train on the uploaded data, validate on the last 28 days and forecast.

    Random Forest is compared with a seasonal naive baseline (same day last week).
    The better method on the validation period is used for the forecast.
    Returns (forecast DataFrame, metrics dictionary).
    """
    if len(daily) < MIN_DAYS:
        raise ValueError(f"At least {MIN_DAYS} days of data are required, got {len(daily)}.")

    data = make_daily_features(daily).dropna(subset=DAILY_FEATURES + ["Revenue"])
    data = data.reset_index(drop=True)
    if len(data) < TEST_DAYS + 30:
        raise ValueError("Not enough usable history after creating lag features.")

    train, test = data.iloc[:-TEST_DAYS], data.iloc[-TEST_DAYS:]
    model = _new_model().fit(train[DAILY_FEATURES], train["Revenue"])
    ml_pred = np.clip(model.predict(test[DAILY_FEATURES]), 0, None)
    naive_pred = test["lag_7"].values

    ml_mae = float(np.abs(test["Revenue"].values - ml_pred).mean())
    naive_mae = float(np.abs(test["Revenue"].values - naive_pred).mean())
    use_ml = ml_mae < naive_mae
    resid_std = float(np.std(test["Revenue"].values - (ml_pred if use_ml else naive_pred)))

    final_model = _new_model().fit(data[DAILY_FEATURES], data["Revenue"]) if use_ml else None

    history = daily[["Date", "Revenue"]].copy()
    rows = []
    for _ in range(horizon):
        next_date = history["Date"].iloc[-1] + pd.Timedelta(days=1)
        if use_ml:
            temp = pd.concat(
                [history, pd.DataFrame({"Date": [next_date], "Revenue": [np.nan]})],
                ignore_index=True)
            features = make_daily_features(temp).iloc[[-1]][DAILY_FEATURES]
            pred = float(final_model.predict(features)[0])
        else:
            pred = float(history["Revenue"].iloc[-7])
        pred = max(pred, 0.0)

        history = pd.concat(
            [history, pd.DataFrame({"Date": [next_date], "Revenue": [pred]})],
            ignore_index=True)
        margin = 0.0 if pred == 0 else 1.96 * resid_std
        rows.append({"Date": next_date, "Forecast": pred,
                     "Lower": max(pred - margin, 0.0), "Upper": pred + margin})

    metrics = {
        "Method": "Random Forest" if use_ml else "Seasonal naive (same day last week)",
        "Random Forest MAE": round(ml_mae, 2),
        "Seasonal naive MAE": round(naive_mae, 2),
        "Validation days": TEST_DAYS,
    }
    return pd.DataFrame(rows), metrics