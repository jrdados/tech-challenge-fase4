import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


WINDOW_SIZE = 60


def build_sequences(series: np.ndarray, window: int = WINDOW_SIZE):
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window : i])
        y.append(series[i])
    return np.array(X), np.array(y)


def prepare_data(df: pd.DataFrame, window: int = WINDOW_SIZE, test_ratio: float = 0.15):
    prices = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    n = len(scaled)
    train_end = int(n * (1 - 2 * test_ratio))
    val_end = int(n * (1 - test_ratio))

    train_scaled = scaled[:train_end]
    val_scaled = scaled[train_end - window : val_end]
    test_scaled = scaled[val_end - window :]

    X_train, y_train = build_sequences(train_scaled, window)
    X_val, y_val = build_sequences(val_scaled, window)
    X_test, y_test = build_sequences(test_scaled, window)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def scale_input(prices: list[float], scaler: MinMaxScaler) -> np.ndarray:
    arr = np.array(prices).reshape(-1, 1)
    scaled = scaler.transform(arr)
    return scaled.reshape(1, len(prices), 1)


def inverse_scale(value: np.ndarray, scaler: MinMaxScaler) -> float:
    return float(scaler.inverse_transform([[value]])[0][0])
