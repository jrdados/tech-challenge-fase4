import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

WINDOW_SIZE = 60


def build_sequences(series: np.ndarray, window: int = WINDOW_SIZE):
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window : i])
        y.append(series[i, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_data(df: pd.DataFrame, window: int = WINDOW_SIZE, test_ratio: float = 0.15):
    prices = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    n = len(scaled)
    train_end = int(n * (1 - 2 * test_ratio))
    val_end = int(n * (1 - test_ratio))

    X_train, y_train = build_sequences(scaled[:train_end], window)
    X_val, y_val = build_sequences(scaled[train_end - window : val_end], window)
    X_test, y_test = build_sequences(scaled[val_end - window :], window)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def scale_input(prices: list[float], scaler: MinMaxScaler) -> np.ndarray:
    arr = np.array(prices, dtype=np.float32).reshape(-1, 1)
    scaled = scaler.transform(arr)
    return scaled.reshape(1, len(prices), 1).astype(np.float32)


def inverse_scale(value: float, scaler: MinMaxScaler) -> float:
    return float(scaler.inverse_transform([[value]])[0][0])
