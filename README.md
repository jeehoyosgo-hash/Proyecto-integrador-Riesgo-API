# API de Análisis de Riesgo Financiero v2

**Proyecto Integrador — Teoría del Riesgo**  
**Curso:** Python para Desarrollo de APIs e Inteligencia Artificial  
**Universidad:** Santo Tomás (USTA)

---

## Descripción

API REST construida con **FastAPI + Pydantic v2** que actúa como motor de cálculo para el análisis de riesgo de un portafolio global de 30 activos financieros distribuidos en 4 regiones y 6 sectores. El sistema descarga datos en tiempo real desde Yahoo Finance y FRED, calcula indicadores técnicos, modelos de riesgo, genera señales automatizadas de trading y recomienda portafolios óptimos según el perfil de riesgo del inversionista.

---

## Estructura del proyecto

```
Proyecto-integrador-Riesgo-APIs/
│
├── backend/
│   ├── main.py                  # Servidor FastAPI — 14 endpoints
│   ├── models.py                # Modelos Pydantic
│   ├── requirements.txt
│   └── services/
│       ├── datos.py             # Yahoo Finance — 30 activos globales
│       ├── indicadores.py       # SMA, EMA, RSI, MACD, Bollinger, Estocástico
│       ├── riesgo.py            # Rendimientos, VaR, CVaR, Kupiec
│       ├── portafolio.py        # CAPM, Beta, Frontera de Markowitz
│       ├── macro.py             # Señales de trading + FRED API
│       └── comparacion.py      # Comparación global + Motor de recomendaciones
│
├── frontend/
│   ├── index.html               # Tablero HTML — 8 secciones interactivas
│   └── streamlit_app.py         # Tablero Streamlit
│
├── .env                         # Claves de API (no se sube a GitHub)
├── .env.example
├── .gitignore
└── README.md
```

---

## Portafolio global — 30 activos, 4 regiones, 6 sectores

| Región | Tickers | Sectores |
|---|---|---|
| 🇺🇸 Norteamérica | AAPL, MSFT, GOOGL, JPM, BAC, GS, XOM, CVX, JNJ, PFE, AMZN, WMT, TSLA, F | Tecnología, Financiero, Energía, Salud, Consumo, Automotriz |
| 🇪🇺 Europa | SAP.DE, ASML.AS, HSBA.L, BNP.PA, TTE.PA, BP.L, NOVN.SW, AZN.L | Tecnología, Financiero, Energía, Salud |
| 🌎 LatAm | EC, CIB, PETR4.SA, ITUB4.SA | Energía, Financiero |
| 🌏 Asia | TM, SONY, SSNLF, INFY, SFTBY | Automotriz, Tecnología |

---

## Instalación y ejecución

```bash
git clone https://github.com/mildretha/Proyecto-integrador-Riesgo-APIs.git
cd Proyecto-integrador-Riesgo-APIs

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r backend/requirements.txt

# Configurar clave FRED (opcional — sin clave usa datos de ejemplo)
# Obtener en: https://fred.stlouisfed.org/docs/api/api_key.html
cp .env.example .env
# Editar .env y agregar FRED_API_KEY=tu_clave

cd backend
python main.py
```

Swagger UI: `http://localhost:8000/docs`

---

## Endpoints — v2

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Health check |
| GET | `/activos` | Catálogo con filtros por región/sector/país |
| GET | `/catalogo` | Catálogo completo organizado por región y sector |
| GET | `/activos/{ticker}/precio` | Precio actual en tiempo real |
| GET | `/precios/{ticker}` | Precios históricos (Yahoo Finance) |
| GET | `/rendimientos/{ticker}` | Rendimientos + pruebas de normalidad |
| GET | `/indicadores/{ticker}` | SMA, EMA, RSI, MACD, Bollinger, Estocástico |
| POST | `/var` | VaR y CVaR — 3 métodos + backtesting Kupiec |
| GET | `/capm` | Beta, Alpha y rendimiento esperado CAPM |
| POST | `/frontera-eficiente` | Frontera eficiente de Markowitz |
| GET | `/alertas` | Señales automáticas de trading |
| GET | `/macro` | Indicadores macroeconómicos FRED |
| GET | `/comparar` | Comparación de activos entre regiones/sectores |
| GET | `/recomendar` | Motor de recomendaciones con scoring multifactor |

---

## Motor de recomendaciones

El endpoint `/recomendar` evalúa cada activo con scoring multifactor y sugiere un portafolio óptimo según el perfil de riesgo.

**Perfiles disponibles:**

| Perfil | Sharpe | Técnico | Momentum | Volatilidad |
|---|---|---|---|---|
| 🛡️ Conservador | 30% | 20% | 10% | 40% |
| ⚖️ Moderado | 40% | 25% | 20% | 15% |
| 🚀 Agresivo | 35% | 25% | 35% | 5% |

El portafolio recomendado se propaga automáticamente a VaR y Markowitz en el tablero.

---

## Tablero HTML

Abre `frontend/index.html` directamente en el navegador con el servidor corriendo.

**8 secciones:**
- Dashboard — catálogo global con filtros
- Precios e Indicadores — gráficas de precio, RSI, MACD, Bollinger
- VaR & CVaR — 3 métodos + Kupiec (usa el portafolio activo)
- Markowitz & CAPM — frontera eficiente + Beta (usa el portafolio activo)
- Señales — alertas de trading por región
- Comparar — comparación de activos entre regiones
- Recomendaciones — motor de scoring · genera portafolio activo
- Macroeconómico — indicadores FRED

---

## Tablero Streamlit

```bash
pip install streamlit plotly
cd frontend
streamlit run streamlit_app.py
```

Abre en `http://localhost:8501`

---

## Flujo recomendado

```
1. Recomendaciones (elige perfil)
       ↓ portafolio activo se guarda globalmente
2. VaR & CVaR (usa el portafolio recomendado)
       ↓
3. Markowitz & CAPM (usa el portafolio recomendado)
       ↓
4. Señales (alertas de los activos del portafolio)
```

---

## Ejemplo de uso — comparación global

```
GET /comparar?tickers=AAPL&tickers=SAP.DE&tickers=TM&tickers=EC
```

Retorna métricas comparativas (Sharpe, VaR, momentum, RSI, drawdown) y ranking por Sharpe Ratio.

---

## Preguntas de reflexión

**1. ¿Por qué se usa `ddof=1`?**
Los rendimientos representan una muestra, no la población completa. El estimador muestral divide entre (n-1) para corregir el sesgo hacia abajo.

**2. ¿Ventaja de separar `services/`?**
Cada módulo es testeable independientemente. Si el transporte cambia (REST → gRPC), la lógica de cálculo no cambia.

**3. ¿Qué pasa cuando FastAPI retorna 422?**
Pydantic intercepta antes de que llegue al endpoint. Detecta la violación y genera un `ValidationError` que FastAPI convierte en HTTP 422 con detalle del campo que falló.

**4. ¿Por qué rendimientos logarítmicos?**
Son aditivos en el tiempo y tienen distribución más simétrica que los simples. Facilitan las pruebas estadísticas y evitan rendimientos menores a -100%.

**5. ¿Qué mide el Sharpe Ratio?**
Retorno en exceso sobre la tasa libre de riesgo por unidad de riesgo: `(E(R) - Rf) / σ`. El portafolio de máximo Sharpe es el óptimo en la teoría de Markowitz.

---

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| Framework API | FastAPI 0.111 |
| Validación | Pydantic v2 |
| Servidor | Uvicorn |
| Datos mercado | yfinance |
| Análisis | NumPy, Pandas, SciPy |
| Optimización | SciPy minimize (SLSQP) |
| Tablero web | HTML + Chart.js |
| Tablero análisis | Streamlit + Plotly |
| Variables entorno | python-dotenv |

---

## Referencias

- FastAPI: https://fastapi.tiangolo.com/
- Pydantic: https://docs.pydantic.dev/
- yfinance: https://github.com/ranaroussi/yfinance
- FRED API: https://fred.stlouisfed.org/docs/api/
- Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance.
- Hull, J. (2018). *Options, Futures, and Other Derivatives*.
