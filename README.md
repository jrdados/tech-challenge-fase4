# Tech Challenge Fase 4 — LSTM Stock Prediction

Previsão do preço de fechamento da ação **ITUB4.SA (Itaú Unibanco)** usando redes neurais **LSTM**, com deploy via **FastAPI + Docker**.

## Estrutura

```
tech-challenge-fase4/
├── notebooks/
│   └── 01_lstm_itub4.ipynb       # Pipeline completa: EDA → LSTM → avaliação
├── src/
│   ├── data/collector.py          # Coleta via yfinance
│   ├── model/
│   │   ├── lstm.py                # Definição, treino e avaliação do modelo
│   │   └── preprocessing.py      # Normalização e sequências
│   └── api/
│       ├── main.py                # FastAPI app
│       └── schemas.py             # Schemas Pydantic
├── models/                        # Modelo e scaler salvos (gerados no treino)
├── reports/figures/               # Gráficos gerados no notebook
├── train.py                       # Script de treino via CLI
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Como executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Treinar o modelo

```bash
python train.py --symbol ITUB4.SA --start 2018-01-01 --epochs 100
```

O modelo é salvo em `models/lstm_itub4.keras` e o scaler em `models/lstm_itub4_scaler.joblib`.

### 3. Rodar a API localmente

```bash
uvicorn src.api.main:app --reload --port 8000
```

Acesse a documentação interativa em: http://localhost:8000/docs

### 4. Deploy com Docker

```bash
docker-compose up --build
```

A API ficará disponível em http://localhost:8000

## Endpoints da API

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/` | Informações da API |
| GET | `/health` | Status e modelo carregado |
| POST | `/predict` | Previsão de preços |
| GET | `/metrics/summary` | Métricas de monitoramento |
| GET | `/metrics` | Métricas Prometheus |
| GET | `/docs` | Documentação Swagger |

### Exemplo de requisição `/predict`

```json
POST /predict
{
  "prices": [32.5, 32.8, 33.1, ...],  // mínimo 60 preços históricos
  "steps": 5                            // dias a prever (1-30)
}
```

```json
{
  "symbol": "ITUB4.SA",
  "predictions": [33.42, 33.51, 33.39, 33.47, 33.55],
  "steps": 5
}
```

## Modelo LSTM

| Camada | Configuração |
|--------|-------------|
| Input | 60 timesteps × 1 feature |
| LSTM 1 | 128 unidades, return_sequences=True, Dropout 20% |
| LSTM 2 | 64 unidades, Dropout 20% |
| Dense | 32 unidades, ReLU |
| Output | 1 neurônio (preço normalizado) |

- **Otimizador:** Adam
- **Loss:** MSE
- **EarlyStopping:** paciência de 10 épocas

## Monitoramento

- `/health` — verifica se o modelo está carregado
- `/metrics/summary` — requisições, previsões, tempo médio de resposta, CPU e memória
- `/metrics` — endpoint Prometheus para integração com Grafana
