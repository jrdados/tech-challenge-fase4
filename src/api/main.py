import time
import psutil
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.schemas import PredictRequest, PredictResponse, HealthResponse, MetricsResponse
from src.model.lstm import load_model
from src.model.preprocessing import scale_input, inverse_scale, WINDOW_SIZE

_state: dict = {
    "model": None,
    "scaler": None,
    "total_requests": 0,
    "total_predictions": 0,
    "response_times": [],
    "start_time": time.time(),
}

SYMBOL = "ITUB4.SA"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model, scaler = load_model("lstm_itub4")
        _state["model"] = model
        _state["scaler"] = scaler
        print("Modelo LSTM carregado com sucesso.")
    except Exception as e:
        print(f"Aviso: modelo não carregado — {e}")
    yield


app = FastAPI(
    title="LSTM Stock Prediction API",
    description="API para previsão do preço de fechamento da ação ITUB4 usando LSTM.",
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)


@app.get("/", tags=["Info"])
def root():
    return {
        "name": "LSTM Stock Prediction API",
        "version": "1.0.0",
        "symbol": SYMBOL,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_state["model"] is not None,
    )


@app.get("/metrics/summary", response_model=MetricsResponse, tags=["Monitoring"])
def metrics_summary():
    times = _state["response_times"]
    avg_ms = float(np.mean(times) * 1000) if times else 0.0
    proc = psutil.Process()
    return MetricsResponse(
        total_requests=_state["total_requests"],
        total_predictions=_state["total_predictions"],
        avg_response_time_ms=round(avg_ms, 2),
        uptime_seconds=round(time.time() - _state["start_time"], 1),
        cpu_percent=psutil.cpu_percent(interval=0.1),
        memory_mb=round(proc.memory_info().rss / 1024 / 1024, 1),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")

    _state["total_requests"] += 1
    t0 = time.time()

    model = _state["model"]
    scaler = _state["scaler"]

    prices = request.prices[-WINDOW_SIZE:]
    if len(prices) < WINDOW_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Forneça pelo menos {WINDOW_SIZE} preços históricos.",
        )

    predictions = []
    window = list(prices)

    for _ in range(request.steps):
        X = scale_input(window[-WINDOW_SIZE:], scaler)
        pred_scaled = model.predict(X, verbose=0)[0][0]
        pred_value = inverse_scale(pred_scaled, scaler)
        predictions.append(round(pred_value, 4))
        window.append(pred_value)

    elapsed = time.time() - t0
    _state["response_times"].append(elapsed)
    _state["total_predictions"] += request.steps

    return PredictResponse(
        symbol=SYMBOL,
        predictions=predictions,
        steps=request.steps,
    )
