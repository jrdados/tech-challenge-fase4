from pydantic import BaseModel, field_validator
from typing import Annotated
from annotated_types import Len


class PredictRequest(BaseModel):
    prices: list[float]
    steps: int = 1

    @field_validator("prices")
    @classmethod
    def min_prices(cls, v):
        if len(v) < 60:
            raise ValueError("É necessário pelo menos 60 preços históricos.")
        return v

    @field_validator("steps")
    @classmethod
    def steps_range(cls, v):
        if v < 1 or v > 30:
            raise ValueError("steps deve estar entre 1 e 30.")
        return v


class PredictResponse(BaseModel):
    symbol: str
    predictions: list[float]
    steps: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class MetricsResponse(BaseModel):
    total_requests: int
    total_predictions: int
    avg_response_time_ms: float
    uptime_seconds: float
    cpu_percent: float
    memory_mb: float
