"""Script de treino do modelo LSTM para ITUB4.SA."""
import argparse
from src.data.collector import load_or_download
from src.model.preprocessing import prepare_data, WINDOW_SIZE
from src.model.lstm import build_model, train_model, evaluate_model, save_model


def main(symbol: str = "ITUB4.SA", start: str = "2018-01-01", epochs: int = 100):
    print(f"Baixando dados de {symbol}...")
    df = load_or_download(symbol, start_date=start, cache=True)
    print(f"Dados: {len(df)} registros ({df.index[0].date()} ate {df.index[-1].date()})")

    print("Preparando sequências...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_data(df, window=WINDOW_SIZE)
    print(f"Treino: {len(X_train)} | Val: {len(X_val)} | Teste: {len(X_test)}")

    print("Construindo modelo LSTM (PyTorch)...")
    model = build_model()

    print("Treinando...")
    train_model(model, X_train, y_train, X_val, y_val, epochs=epochs)

    print("Avaliando no conjunto de teste...")
    metrics, y_true, y_pred = evaluate_model(model, X_test, y_test, scaler)
    print(f"MAE:  R$ {metrics['MAE']:.4f}")
    print(f"RMSE: R$ {metrics['RMSE']:.4f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")

    print("Salvando modelo...")
    save_model(model, scaler, name="lstm_itub4")
    print("Modelo salvo em models/lstm_itub4.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="ITUB4.SA")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    main(args.symbol, args.start, args.epochs)
