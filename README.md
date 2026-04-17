# API de Análisis de Riesgo Financiero

**Proyecto Integrador — Teoría del Riesgo**  
**Curso:** Python para Desarrollo de APIs e Inteligencia Artificial  
**Universidad:** Santo Tomás (USTA)  
**Estudiante:** Mildreth Diaz Polo & Jeison Hoyos

---

## Descripción

API REST construida con **FastAPI + Pydantic v2** que actúa como motor de cálculo para el análisis de riesgo de un portafolio de cinco activos financieros. El sistema descarga datos en tiempo real desde Yahoo Finance y FRED, calcula indicadores técnicos, modelos de riesgo y genera señales automatizadas de trading.

---

## Estructura del proyecto

```
Proyecto-integrador-Riesgo-APIs/
│
├── backend/
│   ├── main.py                  # Servidor FastAPI — 10 endpoints
│   ├── models.py                # Modelos Pydantic (validación de datos)
│   ├── requirements.txt         # Dependencias del proyecto
│   └── services/
│       ├── __init__.py
│       ├── datos.py             # Descarga de precios (Yahoo Finance)
│       ├── indicadores.py       # SMA, EMA, RSI, MACD, Bollinger, Estocástico
│       ├── riesgo.py            # Rendimientos, VaR, CVaR, Kupiec
│       ├── portafolio.py        # CAPM, Beta, Frontera de Markowitz
│       └── macro.py             # Señales de trading + FRED API
│
├── .env                         # Claves de API (no se sube a GitHub)
├── .env.example                 # Plantilla de variables de entorno
├── .gitignore
└── README.md
```

---

## Activos del portafolio

| Ticker | Empresa | Sector |
|--------|---------|--------|
| AAPL | Apple Inc. | Tecnología |
| MSFT | Microsoft Corp. | Tecnología |
| GOOGL | Alphabet Inc. | Tecnología |
| AMZN | Amazon.com Inc. | Consumo |
| TSLA | Tesla Inc. | Automotriz |

---

## Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/mildretha/Proyecto-integrador-Riesgo-APIs.git
cd Proyecto-integrador-Riesgo-APIs
```

### 2. Crear entorno virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r backend/requirements.txt
```

### 4. Configurar variables de entorno

```bash
# Copiar plantilla
copy .env.example .env   # Windows
cp .env.example .env     # macOS/Linux

# Editar .env y agregar tu clave de FRED
# Obtén tu clave gratis en: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=tu_clave_aqui
```

### 5. Ejecutar el servidor

```bash
cd backend
python main.py
```

### 6. Abrir la documentación interactiva

```
http://localhost:8000/docs      # Swagger UI
http://localhost:8000/redoc     # ReDoc
```

---

## Endpoints disponibles

| Método | Ruta | Descripción | Estado |
|--------|------|-------------|--------|
| GET | `/` | Health check del servidor | ✅ |
| GET | `/activos` | Lista los activos del portafolio | ✅ |
| GET | `/activos/{ticker}/precio` | Precio actual en tiempo real | ✅ |
| GET | `/precios/{ticker}` | Precios históricos desde Yahoo Finance | ✅ |
| GET | `/rendimientos/{ticker}` | Rendimientos simples y logarítmicos | ✅ |
| GET | `/indicadores/{ticker}` | Indicadores técnicos completos | ✅ |
| POST | `/var` | VaR y CVaR con 3 métodos | ✅ |
| GET | `/capm` | Beta y rendimiento esperado CAPM | ✅ |
| POST | `/frontera-eficiente` | Frontera eficiente de Markowitz | ✅ |
| GET | `/alertas` | Señales automáticas de trading | ✅ |
| GET | `/macro` | Indicadores macroeconómicos FRED | ✅ |

---

## Módulos implementados

### Módulo 1 — Precios históricos
Descarga precios de cierre ajustados desde Yahoo Finance para el período 2022-presente.

```bash
GET /precios/AAPL?fecha_inicio=2022-01-01
```

### Módulo 2 — Indicadores técnicos

Calcula los siguientes indicadores para cualquier activo:

| Indicador | Descripción | Señal |
|-----------|-------------|-------|
| SMA 20, 50, 200 | Media Móvil Simple | Precio sobre/bajo SMA |
| EMA 20, 50 | Media Móvil Exponencial | Golden/Death Cross |
| RSI 14 | Índice de Fuerza Relativa | < 30 compra, > 70 venta |
| MACD | Convergencia/Divergencia | Cruce de líneas |
| Bollinger Bands | Bandas de volatilidad | Precio fuera de bandas |
| Estocástico | Oscilador de momentum | < 20 compra, > 80 venta |

```bash
GET /indicadores/MSFT?fecha_inicio=2022-01-01
```

### Módulo 3 — Rendimientos y propiedades empíricas

Calcula rendimientos simples y logarítmicos con estadísticas descriptivas y pruebas de normalidad.

**Estadísticas calculadas:**
- Media y volatilidad (diaria y anualizada)
- Asimetría y curtosis
- Rendimiento mínimo y máximo

**Pruebas de normalidad:**
- Jarque-Bera: evalúa si los rendimientos son normales
- Shapiro-Wilk: prueba complementaria sobre los últimos 50 días

```bash
GET /rendimientos/AAPL
```

### Módulo 4 — Valor en Riesgo (VaR) y CVaR

Calcula el VaR con tres métodos y complementa con Expected Shortfall (CVaR).

**Métodos implementados:**

| Método | Supuesto | Fórmula |
|--------|----------|---------|
| Histórico | Ninguno | Percentil alpha de rendimientos reales |
| Paramétrico | Distribución normal | μ + z_α × σ |
| Monte Carlo | Normal con parámetros históricos | Simulación de 10,000 escenarios |

**Backtesting de Kupiec:** valida si el modelo subestima o sobreestima el riesgo real.

```json
POST /var
{
  "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
  "pesos": [0.3, 0.25, 0.2, 0.15, 0.1],
  "nivel_confianza": 0.95
}
```

### Módulo 5 — CAPM y riesgo sistemático

Estima la Beta de cada activo mediante regresión lineal contra el S&P 500.

**Fórmula CAPM:**
```
E(Ri) = Rf + Beta_i × (E(Rm) - Rf)
```

**Métricas calculadas por activo:**
- Beta (riesgo sistemático)
- Alpha (retorno en exceso sobre CAPM)
- R² (proporción de varianza explicada por el mercado)
- Riesgo sistemático vs. idiosincrático

```bash
GET /capm?tasa_libre_riesgo=0.0525
```

### Módulo 6 — Frontera eficiente de Markowitz

Construye la frontera eficiente y determina los portafolios óptimos.

**Portafolios calculados:**

| Portafolio | Objetivo |
|------------|---------|
| Mínima varianza | Minimiza la volatilidad del portafolio |
| Máximo Sharpe | Maximiza el retorno por unidad de riesgo |
| Igual ponderado | Benchmark de referencia (20% cada activo) |

También incluye simulación de 500 portafolios aleatorios para visualizar la nube de Markowitz.

```json
POST /frontera-eficiente
{
  "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
  "pesos": [0.3, 0.25, 0.2, 0.15, 0.1]
}
```

### Módulo 7 — Señales y alertas automatizadas

Genera señales de compra/venta en tiempo real combinando cinco indicadores técnicos.

**Lógica de señales:**

| Indicador | Señal COMPRA | Señal VENTA |
|-----------|-------------|------------|
| RSI | < 30 (sobrevendido) | > 70 (sobrecomprado) |
| MACD | MACD > Signal Line | MACD < Signal Line |
| Bollinger | Precio bajo banda inferior | Precio sobre banda superior |
| EMA Cross | EMA20 > EMA50 | EMA20 < EMA50 |
| Estocástico | %K < 20 | %K > 80 |

Cada señal tiene una fuerza: **FUERTE**, **MODERADA** o **DÉBIL**.
La señal neta resume el consenso: **COMPRA**, **VENTA** o **NEUTRAL**.

```bash
GET /alertas
```

### Módulo 8 — Datos macroeconómicos (FRED)

Consume la API oficial de la Reserva Federal para obtener indicadores macro.

| Serie | Descripción | Uso |
|-------|-------------|-----|
| DGS3MO | T-Bills 3 meses | Tasa libre de riesgo en CAPM |
| DGS10 | Tesoro 10 años | Referencia de largo plazo |
| CPIAUCSL | Inflación (CPI) | Contexto macroeconómico |
| UNRATE | Desempleo | Salud de la economía |
| FEDFUNDS | Tasa de la Fed | Política monetaria |
| VIXCLS | VIX | Volatilidad del mercado |

```bash
GET /macro
```

---

## Ejemplo de respuesta — VaR

```json
{
  "var_historico": {
    "var_decimal": -0.0234,
    "var_porcentaje": "2.34%",
    "var_monetario": -2340.0,
    "cvar_decimal": -0.0312,
    "interpretacion": "Con 95% de confianza, la pérdida máxima diaria no excederá $2,340.00 USD (2.34% del portafolio)"
  },
  "backtesting_kupiec": {
    "modelo_adecuado": true,
    "interpretacion": "Modelo adecuado — excedencias reales coinciden con las esperadas"
  }
}
```

---

## Ejemplo de respuesta — Señales

```json
{
  "total_alertas": 19,
  "alertas_compra": 7,
  "alertas_venta": 12,
  "resumen": {
    "MSFT": {
      "rsi_actual": 70.78,
      "señal_neta": "VENTA",
      "alertas_compra": 1,
      "alertas_venta": 4
    }
  }
}
```

---

## Validación con Pydantic

Todos los datos de entrada son validados automáticamente. Ejemplo — si los pesos no suman 1.0:

```json
{
  "detail": [
    {
      "type": "value_error",
      "msg": "Los pesos deben sumar 1.0. Actualmente suman 0.8000"
    }
  ]
}
```

---

## Stack tecnológico

| Componente | Tecnología |
|------------|------------|
| Framework API | FastAPI 0.111 |
| Validación | Pydantic v2 |
| Servidor ASGI | Uvicorn |
| Datos de mercado | yfinance 0.2+ |
| Análisis numérico | NumPy, Pandas |
| Estadística | SciPy |
| Optimización | SciPy (minimize) |
| Variables de entorno | python-dotenv |
| HTTP | requests |

---

## Preguntas de reflexión

**1. ¿Por qué se usa `ddof=1` en varianza y desviación estándar?**

Se usa `ddof=1` porque los rendimientos representan una **muestra** de los retornos posibles, no la población completa. El estimador muestral divide entre (n-1) en lugar de n, corrigiendo el sesgo hacia abajo en la estimación de la variabilidad real.

**2. ¿Qué ventaja tiene separar `services/` de `main.py`?**

La separación de responsabilidades permite que cada módulo (indicadores, riesgo, portafolio) sea testeable de forma independiente sin levantar el servidor. Si el transporte cambia (de REST a gRPC), la lógica de cálculo no cambia. También facilita la reutilización en notebooks y scripts.

**3. ¿Qué ocurre internamente cuando FastAPI retorna un error 422?**

Pydantic intercepta la petición antes de que llegue al endpoint. Al detectar que un campo viola una restricción (pesos que no suman 1.0, nivel de confianza fuera de rango, etc.), genera un `ValidationError` que FastAPI convierte en una respuesta HTTP 422 con un JSON detallado. El código del endpoint nunca se ejecuta.

**4. ¿Por qué se prefieren los rendimientos logarítmicos en finanzas?**

Los rendimientos logarítmicos son aditivos en el tiempo: el retorno de dos días es la suma de los retornos diarios. Además tienen distribución más simétrica que los retornos simples y facilitan las pruebas estadísticas. La fórmula `ln(P_t/P_{t-1})` evita rendimientos menores a -100%.

**5. ¿Qué mide el Sharpe Ratio y por qué importa?**

El Sharpe Ratio mide el retorno en exceso sobre la tasa libre de riesgo por unidad de riesgo: `(E(R) - Rf) / σ`. Un Sharpe alto significa que el portafolio compensa bien el riesgo asumido. El portafolio de máximo Sharpe es el óptimo en la teoría de Markowitz.

---

## Referencias

- FastAPI: https://fastapi.tiangolo.com/
- Pydantic: https://docs.pydantic.dev/
- NumPy: https://numpy.org/doc/
- yfinance: https://github.com/ranaroussi/yfinance
- FRED API: https://fred.stlouisfed.org/docs/api/
- Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance.
- Hull, J. (2018). *Options, Futures, and Other Derivatives*. Pearson.

---

## Documentación interactiva

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
