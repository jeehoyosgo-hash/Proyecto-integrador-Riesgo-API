# Proyecto Integrador — Teoría del Riesgo + Python para APIs e IA

**Universidad Santo Tomás | Semestre 2026-I**  
**Profesor:** Javier Mauricio Sierra  
**Autores:** _(nombre del equipo)_

---

## Descripción

Sistema integral de análisis de riesgo financiero organizado en **5 capas**:

| Capa | Contenido |
|------|-----------|
| 1 – Datos | APIs externas (yfinance, FRED) + SQLite via SQLAlchemy |
| 2 – Riesgo clásico | Indicadores, rendimientos, EWMA/GARCH, CAPM, VaR+Kupiec, Markowitz QP |
| 3 – Renta fija y derivados | Nelson-Siegel, duración, convexidad, Black-Scholes, stress testing |
| 4 – Machine Learning | Pipeline RandomForest → joblib → Singleton → /predict |
| 5 – Infraestructura | pytest + Docker multi-stage + Render + GitHub Actions CI |

**Stack:** Python 3.11.9 · FastAPI · Pydantic v2 · SQLAlchemy · scikit-learn · Docker · Render

---

## Activos del portafolio

| Ticker | Nombre | Sector | País |
|--------|--------|--------|------|
| AAPL | Apple Inc. | Tecnología | EE.UU. |
| MSFT | Microsoft Corp. | Tecnología | EE.UU. |
| JPM | JPMorgan Chase | Financiero | EE.UU. |
| EC | Ecopetrol S.A. | Energía | Colombia |
| NOVN.SW | Novartis AG | Salud | Suiza |

**Justificación:** diversificación por sector (Tec/Fin/Energía/Salud) y geografía (EE.UU./LatAm/Europa). EC provee exposición al mercado colombiano relevante para el contexto académico.

---

## Instalación local

```bash
# 1. Clonar el repositorio
git clone https://github.com/mildretha/Proyecto-integrador-Riesgo-APIs
cd Proyecto-integrador-Riesgo-APIs

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate.bat     # Windows

# 3. Instalar dependencias
pip install -r backend/requirements.txt

# 4. Configurar variables de entorno
cp backend/.env.example backend/.env
# Editar backend/.env con tus API keys
```

---

## Variables de entorno

Archivo: `backend/.env` (no subir al repositorio — ya está en `.gitignore`)

| Variable | Descripción | Cómo obtenerla |
|----------|-------------|----------------|
| `FRED_API_KEY` | API de FRED (curva de tesoros, Rf) | [fred.stlouisfed.org/docs/api/](https://fred.stlouisfed.org/docs/api/) |
| `ALPHAVANTAGE_API_KEY` | Precios adicionales | [alphavantage.co](https://www.alphavantage.co/support/#api-key) |
| `DATABASE_URL` | Ruta de SQLite | `sqlite:///./riesgo.db` (por defecto) |

Ver `backend/.env.example` para la plantilla completa.

---

## Ejecución del backend

```bash
# Opción A: local con uvicorn
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Opción B: Docker Compose (recomendado)
docker compose up --build

# Documentación automática:
#   http://localhost:8000/docs    (Swagger UI)
#   http://localhost:8000/redoc  (ReDoc)
```

---

## Ejecución del frontend

```bash
cd frontend
streamlit run app.py
# Acceder en: http://localhost:8501
```

---

## Entrenamiento del modelo ML

```bash
cd backend
python -m app.ml.train --ticker AAPL --periodo 3y
# El modelo queda en: backend/app/ml/model.joblib
```

**Propósito analítico:** Clasificación de régimen de mercado (alcista/lateral/bajista) usando 7 features derivadas de los módulos de riesgo (RSI, MACD, EWMA, rendimientos, %B Bollinger, Estocástico).

---

## Ejecución de tests

```bash
cd backend
pytest tests/ -v --tb=short
```

Tests incluidos:
1. RSI sobre serie sintética conocida
2. VaR paramétrico vs. valor analítico teórico
3. Paridad put-call de Black-Scholes
4. Kupiec LR_POF: modelo perfecto no rechaza H0
5. Kupiec LR_POF: modelo malo rechaza H0
6. Nelson-Siegel: RMSE < 0.5pp
7. Duración de Macaulay de bono a la par
8. GET /precios/AAPL retorna 200 con schema correcto
9. POST /var con pesos ≠ 1 retorna HTTP 422

---

## Deploy en Render

**URL backend:** `https://proyecto-riesgo.onrender.com` _(actualizar con URL real)_  
**Swagger UI:** `https://proyecto-riesgo.onrender.com/docs`  
**ReDoc:** `https://proyecto-riesgo.onrender.com/redoc`

> ⚠️ Render free-tier duerme el servicio tras 15 min sin tráfico. Cold start ~30s.  
> Hacer una llamada de calentamiento antes de la sustentación: `curl https://proyecto-riesgo.onrender.com/`

---

## Uso de herramientas de IA

Durante el desarrollo se usó Claude (Anthropic) como asistente para:
- Revisar la implementación del estadístico de Kupiec (LR_POF formal)
- Verificar las fórmulas de Nelson-Siegel y las Greeks de Black-Scholes
- Estructurar el patrón Singleton del modelo ML
- Generar los tests unitarios de referencia

Todo el código fue revisado, entendido y adaptado por el equipo. Las decisiones metodológicas (selección de activos, modelo GARCH elegido, propósito del ML) son originales del equipo.

---

## Estructura del proyecto

```
proyecto-riesgo/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app, todos los endpoints
│   │   ├── config.py            # BaseSettings, .env
│   │   ├── dependencies.py      # Depends(): servicios, predictor
│   │   ├── models.py            # Pydantic schemas (request/response)
│   │   ├── services/
│   │   │   ├── datos.py         # Descarga yfinance, catálogo
│   │   │   ├── indicadores.py   # SMA, EMA, RSI, MACD, Bollinger, Estocástico
│   │   │   ├── riesgo.py        # VaR/CVaR básico
│   │   │   ├── riesgo_completo.py # Kupiec formal, BS, NS, stress
│   │   │   ├── portafolio.py    # CAPM, Markowitz QP
│   │   │   ├── macro.py         # FRED API, alertas
│   │   │   └── comparacion.py   # Comparación multi-activo
│   │   └── ml/
│   │       ├── train.py         # Entrenamiento offline
│   │       ├── predictor.py     # Singleton
│   │       └── model.joblib     # Modelo serializado (generado por train.py)
│   ├── tests/
│   │   └── test_proyecto.py     # Suite de tests (9 tests)
│   ├── Dockerfile               # Multi-stage slim-bookworm
│   ├── requirements.txt         # Versiones fijas
│   └── .env.example
├── frontend/
│   └── app.py                   # Streamlit
├── .github/
│   └── workflows/ci.yml         # GitHub Actions
├── docker-compose.yml
├── .gitignore
└── README.md
```
