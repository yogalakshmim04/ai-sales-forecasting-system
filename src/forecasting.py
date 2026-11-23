import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.config import (
    MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, DAILY_SALES_PATH, WEEKLY_SALES_PATH,
)
from src.feature_engineering import (
    DAILY_FEATURES, WEEKLY_FEATURES,
    make_daily_features, make_weekly_features,
    get_daily_dataset, get_weekly_dataset,
)
from src.baseline import daily_baselines, weekly_baselines
from src.evaluation import compare_models


def get_models():
    """Return fresh (untrained) ML models."""
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=6, min_samples_leaf=3, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE),
    }


def get_feature_importance(model):
    """Return normalized feature importance values for any supported model."""
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    else:
        values = np.abs(model.named_steps["linearregression"].coef_)
    total = values.sum()
    return values / total if total else values


def _config(level):
    """Settings that differ between daily and weekly forecasting.

    ratio_base: if set, the models predict Revenue / ratio_base and the prediction
    is multiplied back by ratio_base.
    window: number of recent periods used by the moving-average baseline.
    """
    if level == "daily":
        return dict(get_dataset=get_daily_dataset, features=DAILY_FEATURES,
                    make=make_daily_features, date_col="Date", raw_path=DAILY_SALES_PATH,
                    step=pd.Timedelta(days=1), baselines=daily_baselines,
                    ratio_base=None, window=7)
    return dict(get_dataset=get_weekly_dataset, features=WEEKLY_FEATURES,
                make=make_weekly_features, date_col="Week", raw_path=WEEKLY_SALES_PATH,
                step=pd.Timedelta(days=7), baselines=weekly_baselines,
                ratio_base="rolling_mean_4", window=4)


def _target(frame, ratio_base):
    if ratio_base:
        return frame["Revenue"] / frame[ratio_base]
    return frame["Revenue"]


def _to_revenue(raw, frame, ratio_base):
    if ratio_base:
        return raw * frame[ratio_base].values
    return raw


def train_level(level):
    """Train all models, compare them with baselines and save the forecasting method.

    If no ML model beats the best baseline, the baseline is used for future forecasts.
    """
    cfg = _config(level)
    feats, date_col, ratio_base = cfg["features"], cfg["date_col"], cfg["ratio_base"]
    train, test = cfg["get_dataset"]()

    X_train, X_test = train[feats], test[feats]
    y_train, y_test = _target(train, ratio_base), test["Revenue"]

    results = cfg["baselines"](test)
    test_preds = {}
    ml_names = []
    importance_rows = []

    for name, model in get_models().items():
        model.fit(X_train, y_train)
        pred = np.clip(_to_revenue(model.predict(X_test), test, ratio_base), 0, None)
        results[name] = (y_test, pred)
        test_preds[name] = pred
        ml_names.append(name)

        for feature, value in zip(feats, get_feature_importance(model)):
            importance_rows.append(
                {"Model": name, "Feature": feature, "Importance": round(float(value), 4)})

    pd.DataFrame(importance_rows).to_csv(
        PROCESSED_DIR / f"{level}_feature_importance.csv", index=False)

    table = compare_models(results)          # sorted by MAE, best first
    best_ml = table[table["Model"].isin(ml_names)].iloc[0]
    best_baseline = table[~table["Model"].isin(ml_names)].iloc[0]
    use_ml = bool(best_ml["MAE"] < best_baseline["MAE"])

    if use_ml:
        method = best_ml["Model"]
        resid_std = float(np.std(y_test.values - test_preds[method]))
        full = pd.concat([train, test])
        final_model = get_models()[method].fit(full[feats], _target(full, ratio_base))
    else:
        method = best_baseline["Model"]
        baseline_pred = np.asarray(results[method][1], dtype=float)
        resid_std = float(np.std(y_test.values - baseline_pred))
        final_model = None

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "name": method, "use_ml": use_ml,
                 "features": feats, "resid_std": resid_std},
                MODELS_DIR / f"{level}_model.pkl")

    table.to_csv(PROCESSED_DIR / f"{level}_model_comparison.csv", index=False)
    pred_df = test[[date_col, "Revenue"]].rename(columns={"Revenue": "Actual"}).reset_index(drop=True)
    for name, p in test_preds.items():
        pred_df[name] = p
    pred_df.to_csv(PROCESSED_DIR / f"{level}_test_predictions.csv", index=False)

    print(f"\n=== {level.upper()} MODEL COMPARISON ===")
    print(table.to_string(index=False))
    print(f"Best ML model      : {best_ml['Model']} (MAE {best_ml['MAE']})")
    print(f"Best baseline      : {best_baseline['Model']} (MAE {best_baseline['MAE']})")
    print(f"Forecast method    : {method}" + ("" if use_ml else "  (baseline beats all ML models)"))
    return table, method


def get_method(level):
    """Name of the method used for future forecasts (shown in the dashboard)."""
    return joblib.load(MODELS_DIR / f"{level}_model.pkl")["name"]


def _baseline_prediction(history, method, window):
    """Next value from the baseline rule that won the comparison."""
    if "Seasonal" in method:
        return float(history.iloc[-7])
    if "Moving Average" in method:
        return float(history.iloc[-window:].mean())
    return float(history.iloc[-1])           # naive


def forecast_future(level, horizon):
    """Recursive forecast: predict one step, feed it back as history, repeat."""
    cfg = _config(level)
    date_col, ratio_base = cfg["date_col"], cfg["ratio_base"]
    bundle = joblib.load(MODELS_DIR / f"{level}_model.pkl")
    model, feats, resid_std = bundle["model"], bundle["features"], bundle["resid_std"]

    df = pd.read_csv(cfg["raw_path"], parse_dates=[date_col])[[date_col, "Revenue"]]

    rows = []
    for _ in range(horizon):
        next_date = df[date_col].iloc[-1] + cfg["step"]

        if bundle["use_ml"]:
            df = pd.concat([df, pd.DataFrame({date_col: [next_date], "Revenue": [np.nan]})],
                           ignore_index=True)
            row = cfg["make"](df).iloc[[-1]]
            pred = float(_to_revenue(model.predict(row[feats]), row, ratio_base)[0])
            df = df.iloc[:-1]
        else:
            pred = _baseline_prediction(df["Revenue"], bundle["name"], cfg["window"])

        pred = max(pred, 0.0)

        # The dataset has no Saturday transactions, so the Saturday forecast is set to 0
        if level == "daily" and next_date.dayofweek == 5:
            pred = 0.0

        df = pd.concat([df, pd.DataFrame({date_col: [next_date], "Revenue": [pred]})],
                       ignore_index=True)
        margin = 0.0 if pred == 0 else 1.96 * resid_std
        rows.append({date_col: next_date, "Forecast": pred,
                     "Lower": max(pred - margin, 0.0), "Upper": pred + margin})

    return pd.DataFrame(rows)


def run_training():
    train_level("daily")
    train_level("weekly")
    print("\n=== NEXT 7 DAYS ===")
    print(forecast_future("daily", 7).round(0).to_string(index=False))
    print("\n=== NEXT 4 WEEKS ===")
    print(forecast_future("weekly", 4).round(0).to_string(index=False))


if __name__ == "__main__":
    run_training()