import numpy as np
from pathlib import Path
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def build_model(window_size: int = 60, units1: int = 128, units2: int = 64, dropout: float = 0.2):
    model = keras.Sequential([
        layers.Input(shape=(window_size, 1)),
        layers.LSTM(units1, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(units2, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs: int = 100, batch_size: int = 32):
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def evaluate_model(model, X_test, y_test, scaler):
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 4)}, y_true, y_pred


def save_model(model, scaler, name: str = "lstm_itub4"):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / f"{name}.keras")
    joblib.dump(scaler, MODELS_DIR / f"{name}_scaler.joblib")


def load_model(name: str = "lstm_itub4"):
    model = keras.models.load_model(MODELS_DIR / f"{name}.keras")
    scaler = joblib.load(MODELS_DIR / f"{name}_scaler.joblib")
    return model, scaler
