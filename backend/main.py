from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uvicorn

from models import PortafolioRequest, HealthCheck

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE LA APLICACIÓN
# ─────────────────────────────────────────────

app = FastAPI(
    title="API de Análisis de Riesgo Financiero",
    description="""
Sistema de análisis de riesgo para portafolios de inversión.

**Módulos disponibles:**
- Precios históricos (Yahoo Finance)
- Rendimientos y estadísticas
- Indicadores técnicos: SMA, EMA, RSI, MACD, Bollinger
- Valor en Riesgo (VaR) y CVaR
- CAPM y Beta
- Frontera eficiente de Markowitz
- Señales de trading automatizadas
- Datos macroeconómicos (FRED)
    """,
    version="1.0.0",
)

# CORS: permite que el tablero (Streamlit) se conecte a esta API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Los 5 activos del portafolio (mínimo requerido por el taller)
ACTIVOS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]


# ─────────────────────────────────────────────
# ENDPOINT 1: HEALTH CHECK — GET /
# ─────────────────────────────────────────────

@app.get("/", response_model=HealthCheck, tags=["Sistema"])
def health_check():
    """
    Verifica que el servidor esté corriendo.
    Es el primer endpoint que debes probar.
    """
    return HealthCheck(
        status="ok",
        mensaje="API de Riesgo Financiero funcionando correctamente",
        version="1.0.0",
        activos_disponibles=ACTIVOS
    )


# ─────────────────────────────────────────────
# ENDPOINT 2: LISTAR ACTIVOS — GET /activos
# ─────────────────────────────────────────────

@app.get("/activos", tags=["Activos"])
def listar_activos():
    """
    Lista los activos del portafolio con información básica.
    En la Fase 3 esto se conectará a Yahoo Finance para precios en tiempo real.
    """
    activos = [
        {"ticker": "AAPL",  "nombre": "Apple Inc.",      "sector": "Tecnología"},
        {"ticker": "MSFT",  "nombre": "Microsoft Corp.", "sector": "Tecnología"},
        {"ticker": "GOOGL", "nombre": "Alphabet Inc.",   "sector": "Tecnología"},
        {"ticker": "AMZN",  "nombre": "Amazon.com Inc.", "sector": "Consumo"},
        {"ticker": "TSLA",  "nombre": "Tesla Inc.",      "sector": "Automotriz"},
    ]
    return {"total": len(activos), "activos": activos}


# ─────────────────────────────────────────────
# ENDPOINT 3: PRECIOS — GET /precios/{ticker}
# ─────────────────────────────────────────────

@app.get("/precios/{ticker}", tags=["Precios"])
def obtener_precios(
    ticker: str,
    fecha_inicio: str = Query(default="2022-01-01", description="Formato YYYY-MM-DD"),
    fecha_fin: Optional[str] = Query(default=None, description="Formato YYYY-MM-DD"),
):
    """
    Retorna precios históricos de cierre de un activo.
    Se conectará a Yahoo Finance en la Fase 3.
    """
    ticker = ticker.upper()
    if ticker not in ACTIVOS:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' no encontrado. Disponibles: {ACTIVOS}"
        )
    return {
        "ticker": ticker,
        "fecha_inicio": fecha_inicio,
        "fecha_fin": fecha_fin or "hoy",
        "nota": "Datos reales en Fase 3",
        "muestra": [
            {"fecha": "2024-01-02", "cierre": 185.20},
            {"fecha": "2024-01-03", "cierre": 184.25},
        ],
    }


# ─────────────────────────────────────────────
# ENDPOINT 4: RENDIMIENTOS — GET /rendimientos/{ticker}
# ─────────────────────────────────────────────

@app.get("/rendimientos/{ticker}", tags=["Análisis"])
def obtener_rendimientos(
    ticker: str,
    fecha_inicio: str = Query(default="2022-01-01"),
):
    """
    Calcula rendimientos simples y logarítmicos.
    Se implementará en la Fase 5.
    """
    ticker = ticker.upper()
    if ticker not in ACTIVOS:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no encontrado")
    return {
        "ticker": ticker,
        "nota": "Cálculo real en Fase 5",
        "estadisticas_ejemplo": {
            "media_diaria":  0.0012,
            "desv_estandar": 0.0189,
            "minimo":       -0.0892,
            "maximo":        0.0756,
            "asimetria":    -0.23,
            "curtosis":      3.8,
        },
    }


# ─────────────────────────────────────────────
# ENDPOINT 5: INDICADORES — GET /indicadores/{ticker}
# ─────────────────────────────────────────────

@app.get("/indicadores/{ticker}", tags=["Análisis"])
def obtener_indicadores(ticker: str):
    """
    Retorna indicadores técnicos: SMA, EMA, RSI, MACD, Bollinger Bands.
    Se implementará en la Fase 4.
    """
    ticker = ticker.upper()
    if ticker not in ACTIVOS:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no encontrado")
    return {
        "ticker": ticker,
        "indicadores_disponibles": ["SMA_20", "SMA_50", "EMA_20", "RSI_14", "MACD", "Bollinger"],
        "nota": "Valores reales en Fase 4",
    }


# ─────────────────────────────────────────────
# ENDPOINT 6: VaR — POST /var
# ─────────────────────────────────────────────

@app.post("/var", tags=["Riesgo"])
def calcular_var(portafolio: PortafolioRequest):
    """
    Calcula el Valor en Riesgo (VaR) y CVaR del portafolio.

    Pydantic valida automáticamente que los pesos sumen 1.0.
    Se implementará en la Fase 5.
    """
    return {
        "tickers":          portafolio.tickers,
        "pesos":            portafolio.pesos,
        "nivel_confianza":  portafolio.nivel_confianza,
        "nota":             "Cálculo real en Fase 5",
        "resultado_ejemplo": {
            "var_historico":   -0.0234,
            "var_parametrico": -0.0198,
            "cvar":            -0.0312,
            "interpretacion":  (
                f"Con {portafolio.nivel_confianza*100:.0f}% de confianza, "
                "la pérdida máxima diaria no excedería el 2.34%"
            ),
        },
    }


# ─────────────────────────────────────────────
# ENDPOINT 7: CAPM — GET /capm
# ─────────────────────────────────────────────

@app.get("/capm", tags=["Riesgo"])
def calcular_capm(
    tickers: List[str] = Query(default=ACTIVOS)
):
    """
    Calcula Beta y rendimiento esperado por CAPM para cada activo.
    Fórmula: E(Ri) = Rf + Beta × (E(Rm) - Rf)
    Se implementará en la Fase 6.
    """
    return {
        "benchmark": "S&P 500 (^GSPC)",
        "nota":      "Cálculo real en Fase 6",
        "resultado_ejemplo": {
            "AAPL":  {"beta": 1.23, "rendimiento_esperado_anual": "14.2%"},
            "MSFT":  {"beta": 1.15, "rendimiento_esperado_anual": "13.5%"},
            "GOOGL": {"beta": 1.08, "rendimiento_esperado_anual": "12.8%"},
        },
    }


# ─────────────────────────────────────────────
# ENDPOINT 8: FRONTERA EFICIENTE — POST /frontera-eficiente
# ─────────────────────────────────────────────

@app.post("/frontera-eficiente", tags=["Portafolio"])
def calcular_frontera(portafolio: PortafolioRequest):
    """
    Construye la frontera eficiente de Markowitz.
    Determina el portafolio de mínima varianza y el de máximo Sharpe Ratio.
    Se implementará en la Fase 6.
    """
    return {
        "tickers": portafolio.tickers,
        "nota":    "Frontera eficiente en Fase 6",
        "resultado_ejemplo": {
            "portafolio_min_varianza": {
                "pesos":                [0.30, 0.25, 0.20, 0.15, 0.10],
                "rendimiento_esperado": "11.8%",
                "volatilidad":          "16.2%",
                "sharpe_ratio":         0.73,
            },
        },
    }


# ─────────────────────────────────────────────
# ENDPOINT 9: ALERTAS — GET /alertas
# ─────────────────────────────────────────────

@app.get("/alertas", tags=["Señales"])
def obtener_alertas():
    """
    Señales automáticas de compra/venta basadas en indicadores técnicos.
    RSI < 30 → COMPRA. RSI > 70 → VENTA.
    Se implementará en la Fase 7.
    """
    return {
        "total_alertas": 0,
        "alertas":       [],
        "nota":          "Señales reales en Fase 7",
    }


# ─────────────────────────────────────────────
# ENDPOINT 10: DATOS MACRO — GET /macro
# ─────────────────────────────────────────────

@app.get("/macro", tags=["Macro"])
def obtener_macro():
    """
    Indicadores macroeconómicos desde FRED API:
    tasa libre de riesgo, inflación, desempleo.
    Se conectará a FRED en la Fase 7.
    """
    return {
        "fuente": "FRED - Federal Reserve Bank of St. Louis",
        "nota":   "Conexión real a FRED en Fase 7",
        "datos_ejemplo": {
            "tasa_libre_riesgo": 0.0525,
            "inflacion_anual":   0.033,
            "tasa_desempleo":    0.039,
        },
    }


# ─────────────────────────────────────────────
# ARRANCAR EL SERVIDOR
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
