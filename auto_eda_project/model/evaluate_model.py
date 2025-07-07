'''
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_model(model, X, y, label="Test"):
    preds = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    print(f"\n📊 {label} Set Evaluation:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")

    return {"rmse": rmse, "mae": mae, "r2": r2}
'''
###########
'''
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_model(model, X, y_log, label="Test"):
    """
    Evaluates the model on the test set with log-transformed target.
    Applies np.expm1() to get back to original scale before metrics.

    Parameters:
        model: Trained regression model
        X: Features (untransformed)
        y_log: Log-transformed target (i.e., np.log1p(y))
        label: Evaluation set label ("Test", "Validation", etc.)

    Returns:
        dict: {"rmse": ..., "mae": ..., "r2": ...}
    """
    # Predict in log-space
    y_pred_log = model.predict(X)
    # 🔍 Sanity check to inspect scales
    print("\n🧪 Sanity check:")
    print("y_log (true, log scale):", y_log[:5])
    print("y (true, original scale):", np.expm1(y_log[:5]))
    print("y_pred_log (predicted, log scale):", y_pred_log[:5])
    print("y_pred (predicted, original scale):", np.expm1(y_pred_log[:5]))

    # Convert back to original scale
    y = np.expm1(y_log)
    y_pred = np.expm1(y_pred_log)

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"\n📊 {label} Set Evaluation:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")

    return {"rmse": rmse, "mae": mae, "r2": r2}
'''
###########################
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    """Compute MAPE with zero handling."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100

def evaluate_model(model, X, y_log, label="Test"):
    y_pred_log = model.predict(X)

    y_true = np.expm1(y_log)
    y_pred = np.expm1(y_pred_log)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print(f"\n📊 {label} Set Evaluation:")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f} | MAPE: {mape:.2f}%")

    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}
