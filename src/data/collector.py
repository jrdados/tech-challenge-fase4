import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def download_stock_data(
    symbol: str = "ITUB4.SA",
    start_date: str = "2018-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close", "Open", "High", "Low", "Volume"]].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(inplace=True)
    return df


def load_or_download(
    symbol: str = "ITUB4.SA",
    start_date: str = "2018-01-01",
    end_date: str | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    cache_path = DATA_DIR / f"{symbol.replace('.', '_')}.parquet"
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)
    df = download_stock_data(symbol, start_date, end_date)
    if cache:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
    return df
