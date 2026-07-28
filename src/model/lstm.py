import numpy as np
from pathlib import Path
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden1=128, hidden2=64, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])
        out = self.relu(self.fc1(out))
        return self.fc2(out)


def build_model(hidden1=128, hidden2=64, dropout=0.2):
    return LSTMModel(hidden1=hidden1, hidden2=hidden2, dropout=dropout).to(DEVICE)


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, lr=1e-3, patience=10):
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_v = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)
    criterion = nn.MSELoss()

    history = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_mae = 0.0, 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            train_mae += torch.mean(torch.abs(pred - yb)).item() * len(xb)

        train_loss /= len(X_tr)
        train_mae /= len(X_tr)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, y_v).item()
            val_mae = torch.mean(torch.abs(val_pred - y_v)).item()

        scheduler.step(val_loss)
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | loss={train_loss:.6f} | val_loss={val_loss:.6f} | val_mae={val_mae:.6f}")

        if no_improve >= patience:
            print(f"EarlyStopping na época {epoch}.")
            break

    if best_state:
        model.load_state_dict(best_state)
    return history


def evaluate_model(model, X_test, y_test, scaler):
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        y_pred_scaled = model(X_t).cpu().numpy()

    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 4)}, y_true, y_pred


def save_model(model, scaler, name="lstm_itub4"):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / f"{name}.pt")
    joblib.dump(scaler, MODELS_DIR / f"{name}_scaler.joblib")


def load_model(name="lstm_itub4"):
    model = LSTMModel().to(DEVICE)
    model.load_state_dict(torch.load(MODELS_DIR / f"{name}.pt", map_location=DEVICE))
    model.eval()
    scaler = joblib.load(MODELS_DIR / f"{name}_scaler.joblib")
    return model, scaler


def predict_one(model, X_input: np.ndarray) -> float:
    model.eval()
    t = torch.tensor(X_input, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        return model(t).cpu().item()
