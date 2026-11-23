# Model evaluation
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(y_true, y_pred):
    """MAE, RMSE, R2 and WAPE for one model."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # WAPE = total error / total actual sales (MAPE-ku safe alternative)
    wape = np.abs(y_true - y_pred).sum() / np.abs(y_true).sum() * 100

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 3),
        "WAPE_%": round(wape, 2),
    }


def compare_models(results):
    """results = {"Model name": (y_true, y_pred), ...} -> comparison table sorted by MAE."""
    rows = [{"Model": name, **evaluate(t, p)} for name, (t, p) in results.items()]
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)