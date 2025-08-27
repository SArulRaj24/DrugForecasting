# FILE: backend/models/arima_model.py
from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")


def train_best_arima(y: pd.Series) -> Tuple[Any, Dict[str, Any], PowerTransformer]:
    """
    Grid-search ARIMA with Yeo-Johnson transform.
    Returns best model, config, and transformer.
    """
    assert isinstance(y.index, pd.DatetimeIndex)

    # Apply Yeo-Johnson transform
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    y_trans = pt.fit_transform(y.values.reshape(-1, 1)).flatten()
    y_trans_series = pd.Series(y_trans, index=y.index)

    best_model, best_aic, best_order = None, np.inf, None
    p_range, d_range, q_range = range(0, 4), range(0, 2), range(0, 4)

    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(y_trans_series, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_model, best_aic, best_order = fitted, fitted.aic, (p, d, q)
                except Exception:
                    continue

    if best_model is None:
        raise RuntimeError("Failed to fit ARIMA model on the provided data.")

    return best_model, {"order": best_order, "aic": float(best_aic)}, pt


def forecast_with_model(model, transformer: PowerTransformer, horizon: int) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Forecast in original scale.
    horizon = number of steps ahead (respects input frequency)
    """
    fc_trans = model.forecast(steps=horizon)
    if isinstance(fc_trans, np.ndarray):
        fc_trans = pd.Series(fc_trans)

    fc_orig = transformer.inverse_transform(fc_trans.values.reshape(-1, 1)).flatten()

    # Find last date in the series
    if hasattr(model.data, "row_labels") and model.data.row_labels is not None:
        last = model.data.row_labels[-1]
    else:
        last = model.model.endog.index[-1]
    last = pd.to_datetime(last)

    # Detect frequency
    freq = pd.infer_freq(model.data.row_labels)
    if freq is None:
        raise ValueError("Could not infer frequency. Please specify it manually.")

    # Generate future dates with correct frequency
    idx = pd.date_range(last, periods=horizon+1, freq=freq)[1:]
    return idx, fc_orig



def evaluate_model(y: pd.Series, model, transformer: PowerTransformer) -> Dict[str, float]:
    """Evaluate model fit (MAE, RMSE)."""
    fitted_trans = model.fittedvalues
    if isinstance(fitted_trans, np.ndarray):
        fitted_trans = pd.Series(fitted_trans, index=y.index)

    fitted_orig = transformer.inverse_transform(fitted_trans.values.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y, fitted_orig)
    rmse = np.sqrt(mean_squared_error(y, fitted_orig))
    return {"mae": float(mae), "rmse": float(rmse)}
