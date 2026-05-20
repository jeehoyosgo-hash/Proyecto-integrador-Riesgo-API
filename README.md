# Proyecto Integrador — Teoría del Riesgo + Python para APIs e IA

**Universidad Santo Tomás | Semestre 2026-I**  
**Profesor:** Javier Mauricio Sierra  
**Autores:** *(completar con nombres reales del equipo)*

---

## Descripción

Sistema integral de análisis de riesgo financiero con arquitectura separada backend/frontend:

| Capa | Contenido |
|------|-----------|
| 1 – Datos | Yahoo Finance + FRED API + fallback sintético + SQLite via SQLAlchemy |
| 2 – Riesgo clásico | Indicadores técnicos, rendimientos, EWMA/GARCH, CAPM, VaR+Kupiec, Markowitz QP |
| 3 – Renta fija y derivados | Nelson-Siegel, duración/convexidad, Black-Scholes, stress testing |
| 4 – Machine Learning | Pipeline RandomForest → joblib → Singleton → /predict |
| 5 – Infraestructura | pytest + Docker multi-stage + Render + GitHub Actions CI |

**Stack:** Python 3.11 · FastAPI · Pydantic v2 · SQLAlchemy · scikit-learn · Docker · Render

---

## Portafolio de activos

El portafolio está compuesto por **10 activos de 4 regiones y 5 sectores**, garantizando diversificación real.

| Ticker | Nombre | Sector | País | Región |
|--------|--------|--------|------|--------|
| AAPL | Apple Inc. | Tecnología | EE.UU. | Norteamérica |
| MSFT | Microsoft Corp. | Tecnología | EE.UU. | Norteamérica |
| JPM | JPMorgan Chase | Financiero | EE.UU. | Norteamérica |
| XOM | ExxonMobil Corp. | Energía | EE.UU. | Norteamérica |
| JNJ | Johnson & Johnson | Salud | EE.UU. | Norteamérica |
| SAP.DE | SAP SE | Tecnología | Alemania | Europa |
| NOVN.SW | Novartis AG | Salud | Suiza | Europa |
| EC | Ecopetrol S.A. | Energía | Colombia | LatAm |
| CIB | Bancolombia S.A. | Financiero | Colombia | LatAm |
| TM | Toyota Motor Corp. | Automotriz | Japón | Asia |

**Justificación de la selección:**
- **Diversificación sectorial:** Tecnología, Financiero, Energía, Salud, Automotriz — 5 sectores con correlaciones bajas entre sí.
- **Diversificación geográfica:** 4 regiones (Norteamérica, Europa, LatAm, Asia) — exposición a distintos ciclos económicos y monedas (USD, EUR, CHF).
- **Contexto académico colombiano:** EC (Ecopetrol) y CIB (Bancolombia) son activos del mercado colombiano, relevantes para el análisis de riesgo en el contexto de la USTA. EC tiene alta correlación con el precio del petróleo — útil para discutir riesgo sistemático.
- **Perfil de riesgo diverso:** JNJ y NOVN.SW son defensivos (beta < 1); AAPL y TSLA son agresivos (beta > 1); JPM es sensible a tasas de interés — conjunto ideal para demostrar CAPM y frontera eficiente.

---

## Instalación local

```bash
# 1. Clonar el repositorio
git clone https://github.com/jeehoyosgo-hash/Proyecto-integrador-Riesgo-API
cd Proyecto-integrador-Riesgo-API

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate.bat     # Windows

# 3. Instalar dependencias del backend
pip install -r backend/requirements.txt

# 4. Configurar variables de entorno
cp backend/.env.example backend/.env
# Editar backend/.env con tus API keys (ver sección siguiente)
```

---

## Variables de entorno

Archivo: `backend/.env` — **nunca subir al repositorio** (está en `.gitignore`)

| Variable | Descripción | Cómo obtenerla |
|----------|-------------|----------------|
| `FRED_API_KEY` | API de FRED (curva de tesoros, Rf, macro) | [fred.stlouisfed.org/docs/api](https://fred.stlouisfed.org/docs/api/) — gratuita |
| `ALPHAVANTAGE_API_KEY` | Precios adicionales | [alphavantage.co](https://www.alphavantage.co/support/#api-key) — gratuita |
| `DATABASE_URL` | Ruta de SQLite | `sqlite:///./riesgo.db` (por defecto, no requiere cambio) |

Ver `backend/.env.example` para la plantilla completa con todos los campos y valores por defecto.

> **Nota:** Si no configuras `FRED_API_KEY`, el sistema usa datos de referencia hardcodeados para la tasa libre de riesgo y los indicadores macro. El tablero funciona correctamente sin keys — ideal para demo sin conexión.

---

## Ejecución del backend

```bash
# Opción A — local con uvicorn
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Opción B — Docker Compose (recomendado para sustentación)
docker compose up --build

# Documentación automática (Swagger UI):
#   http://localhost:8000/docs
#   http://localhost:8000/redoc
```

---

## Ejecución del frontend

```bash
cd frontend
streamlit run app.py
# Acceder en: http://localhost:8501
```

El frontend consume el backend vía HTTP (`http://localhost:8000`). Asegúrate de que el backend esté corriendo antes de iniciar el frontend.

---

## Entrenamiento del modelo ML

```bash
cd backend
python -m app.ml.train --ticker AAPL --periodo 3y
# El modelo se guarda en: backend/app/ml/model.joblib
```

El modelo clasifica el régimen de mercado (alcista/lateral/bajista) usando 7 features derivadas de los módulos de riesgo: RSI, MACD histograma, volatilidad EWMA, rendimiento a 5 y 21 días, %B de Bollinger, y Estocástico %K.

---

## Ejecución de tests

```bash
cd backend
pytest tests/ -v --tb=short

# Con reporte de cobertura:
pytest tests/ -v --cov=app --cov-report=term-missing
```

Tests incluidos (9 en total):
1. RSI sobre serie sintética con valor analítico conocido
2. VaR paramétrico vs. valor analítico teórico
3. Paridad put-call de Black-Scholes
4. Kupiec LR_POF: modelo perfecto no rechaza H0
5. Kupiec LR_POF: modelo malo rechaza H0
6. Nelson-Siegel: RMSE < 0.5 puntos porcentuales
7. Duración Macaulay de bono a la par (debe ser < vencimiento)
8. `GET /precios/AAPL` retorna HTTP 200 con schema correcto
9. `POST /var` con pesos ≠ 1 retorna HTTP 422 (validación Pydantic)

---

## Endpoints principales de la API

La documentación interactiva completa está disponible en `/docs` (Swagger UI) cuando el backend está corriendo.

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/activos` | GET | Lista activos del catálogo con filtros por región/sector |
| `/precios/{ticker}` | GET | Precios históricos OHLCV de un activo |
| `/rendimientos/{ticker}` | GET | Rendimientos simples/log + estadísticas + pruebas de normalidad |
| `/indicadores/{ticker}` | GET | SMA, EMA, RSI, MACD, Bollinger, Estocástico + señales |
| `/volatilidad/{ticker}` | GET | EWMA configurable + ARCH/GARCH + tabla AIC/BIC |
| `/var` | POST | VaR paramétrico, histórico y Monte Carlo + Kupiec |
| `/capm` | GET | Beta y rendimiento esperado CAPM con Rf de FRED |
| `/frontera-eficiente` | POST | Frontera Markowitz QP + mínima varianza + máximo Sharpe |
| `/alertas` | GET | Señales compra/venta por indicador (umbrales configurables) |
| `/macro` | GET | Indicadores FRED: Rf, inflación, desempleo, VIX |
| `/curva-rendimiento` | GET | Curva de tesoros US + Nelson-Siegel |
| `/bono/duracion` | POST | Duración Macaulay, modificada y convexidad |
| `/opcion/precio` | POST | Black-Scholes + 5 Greeks + volatilidad implícita |
| `/stress` | POST | Stress testing: shock tasas, caída mercado, volatilidad ×2 |
| `/predict` | POST | Régimen de mercado ML (alcista/lateral/bajista) |

---

## Estructura del proyecto

```
proyecto-riesgo/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app — todos los endpoints
│   │   ├── config.py            # BaseSettings + variables de entorno
│   │   ├── dependencies.py      # Depends(): servicios inyectados
│   │   ├── models.py            # Pydantic schemas (request/response)
│   │   ├── services/
│   │   │   ├── datos.py         # Descarga yfinance + catálogo de activos
│   │   │   ├── indicadores.py   # SMA, EMA, RSI, MACD, Bollinger, Estocástico
│   │   │   ├── riesgo.py        # Rendimientos, VaR/CVaR básico
│   │   │   ├── riesgo_completo.py # VaR completo, Kupiec, BS, NS, stress
│   │   │   ├── portafolio.py    # CAPM, Markowitz QP
│   │   │   ├── macro.py         # FRED API + señales de trading
│   │   │   └── comparacion.py   # Benchmark, alpha Jensen, tracking error
│   │   └── ml/
│   │       ├── train.py         # Entrenamiento offline del modelo
│   │       ├── predictor.py     # Singleton para carga única del modelo
│   │       └── model.joblib     # Modelo serializado (generar con train.py)
│   ├── tests/
│   │   └── test_proyecto.py     # Suite de 9 tests unitarios
│   ├── Dockerfile               # Multi-stage build (slim-bookworm)
│   ├── requirements.txt         # Dependencias con versiones fijas
│   └── .env.example             # Plantilla de variables de entorno
├── frontend/
│   └── app.py                   # Streamlit (consume el backend vía HTTP)
├── .github/
│   └── workflows/ci.yml         # GitHub Actions: lint + tests en cada push
├── docker-compose.yml           # Orquestación backend + frontend
├── render.yaml                  # Configuración de despliegue en Render
├── .gitignore                   # Excluye .env, __pycache__, model.joblib, etc.
└── README.md
```

---

## Despliegue en Render

**URL backend:** *(actualizar con URL real del despliegue)*  
**Swagger UI:** `<URL_BACKEND>/docs`  
**ReDoc:** `<URL_BACKEND>/redoc`

> ⚠️ Render free-tier duerme el servicio tras 15 min sin tráfico. Cold start ~30s.  
> Antes de la sustentación, hacer una llamada de calentamiento: `curl <URL_BACKEND>/`

---

## Uso de herramientas de IA

Durante el desarrollo se usó **Claude (Anthropic)** como asistente para:

- Revisar la implementación formal del estadístico de Kupiec (LR_POF con distribución chi²)
- Verificar las fórmulas de Nelson-Siegel y las Greeks de Black-Scholes
- Estructurar el patrón Singleton del modelo ML
- Generar los tests unitarios de referencia y verificar casos borde
- Identificar inconsistencias entre el README y el código (portafolio de activos, imports)

Todo el código fue revisado, entendido y adaptado por el equipo. Las decisiones metodológicas —selección de activos, modelo GARCH elegido, umbral de Kupiec, propósito del ML— son originales del equipo.
