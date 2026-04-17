# main.py
# App principal de FastAPI con los 9 endpoints requeridos.
# Cada endpoint usa Depends() — nunca hay lógica directa en las rutas.

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import time
import functools

from app.config import get_settings, Settings
from app.models import PortafolioRequest, RendimientoObjetivoRequest
from app.services import TechnicalIndicators, RiskCalculator, PortfolioAnalyzer
from app.dependencies import (
    get_technical_indicators,
    get_risk_calculator,
    get_portfolio_analyzer,
    get_macro_data,
)

# ── Decorador personalizado: mide tiempo de ejecución ──
def timer_decorator(func):
    """Decorador que registra el tiempo de ejecución de cada endpoint."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        inicio = time.time()
        resultado = await func(*args, **kwargs)
        tiempo = round(time.time() - inicio, 3)
        if isinstance(resultado, dict):
            resultado["tiempo_ejecucion_seg"] = tiempo
        return resultado
    return wrapper


# ── Inicialización de la app ──
app = FastAPI(
    title="🏦 API de Análisis de Riesgo Financiero",
    description="""
    ## Proyecto Integrador — Teoría del Riesgo con Python

    API RESTful para análisis cuantitativo de portafolios financieros.

    ### Portafolio analizado:
    - **NVDA** — NVIDIA (Semiconductores)
    - **BRK-B** — Berkshire Hathaway (Conglomerado)
    - **LIN** — Linde PLC (Química/Industrial)
    - **EC** — Ecopetrol (Energía/Colombia)
    - **GLD** — SPDR Gold (Commodities)
    - **VNQ** — Vanguard Real Estate (REIT)

    ### Módulos disponibles:
    Indicadores técnicos, Rendimientos, GARCH, CAPM, VaR/CVaR, Markowitz, Alertas, Macro
    """,
    version="1.0.0",
    contact={"name": "Proyecto USTA — Teoría del Riesgo"},
)

# CORS: permite que el frontend (Streamlit) consuma la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TICKERS_DEFAULT = ["NVDA", "BRK-B", "LIN", "EC", "GLD", "VNQ"]


# ────────────────────────────────────────────────────────
# ENDPOINT 1: /activos — Lista de activos del portafolio
# ────────────────────────────────────────────────────────
@app.get("/activos", tags=["Portafolio"])
async def listar_activos(
    settings: Settings = Depends(get_settings),
    ti: TechnicalIndicators = Depends(get_technical_indicators),
):
    """
    Retorna información actual de cada activo del portafolio.
    Precio actual, nombre, sector y moneda.
    """
    info_map = {
        "NVDA":  {"nombre": "NVIDIA Corporation",       "sector": "Semiconductores"},
        "BRK-B": {"nombre": "Berkshire Hathaway B",     "sector": "Conglomerado Financiero"},
        "LIN":   {"nombre": "Linde PLC",                "sector": "Química e Industrial"},
        "EC":    {"nombre": "Ecopetrol S.A.",           "sector": "Energía - Colombia"},
        "GLD":   {"nombre": "SPDR Gold Shares ETF",     "sector": "Commodities - Oro"},
        "VNQ":   {"nombre": "Vanguard Real Estate ETF", "sector": "Bienes Raíces (REIT)"},
    }

    activos = []
    for ticker in TICKERS_DEFAULT:
        try:
            df = ti.obtener_precios(ticker, "5d")
            precio = round(float(df["Close"].iloc[-1]), 2)
            activos.append({
                "ticker": ticker,
                "nombre": info_map.get(ticker, {}).get("nombre", ticker),
                "sector": info_map.get(ticker, {}).get("sector", "N/A"),
                "precio_actual": precio,
                "moneda": "USD",
            })
        except Exception as e:
            activos.append({"ticker": ticker, "error": str(e)})

    return {"activos": activos, "total": len(activos)}


# ────────────────────────────────────────────────────────
# ENDPOINT 2: /precios/{ticker} — Precios históricos
# ────────────────────────────────────────────────────────
@app.get("/precios/{ticker}", tags=["Datos"])
async def precios_historicos(
    ticker: str,
    periodo: str = Query(default="2y", description="Período: 1y, 2y, 3y, 5y"),
    ti: TechnicalIndicators = Depends(get_technical_indicators),
):
    """
    Retorna precios históricos OHLCV de un activo.
    Incluye apertura, máximo, mínimo, cierre y volumen.
    """
    ticker = ticker.upper()
    if ticker not in TICKERS_DEFAULT:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' no encontrado. Disponibles: {TICKERS_DEFAULT}"
        )

    try:
        df = ti.obtener_precios(ticker, periodo)
        precios = []
        for fecha, row in df.iterrows():
            precios.append({
                "fecha": str(fecha.date()),
                "apertura": round(float(row.get("Open", 0)), 2),
                "maximo": round(float(row.get("High", 0)), 2),
                "minimo": round(float(row.get("Low", 0)), 2),
                "cierre": round(float(row.get("Close", 0)), 2),
                "volumen": int(row.get("Volume", 0)),
            })
        return {"ticker": ticker, "periodo": periodo, "datos": precios, "n_observaciones": len(precios)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error obteniendo datos: {str(e)}")


# ────────────────────────────────────────────────────────
# ENDPOINT 3: /rendimientos/{ticker}
# ────────────────────────────────────────────────────────
@app.get("/rendimientos/{ticker}", tags=["Análisis"])
async def rendimientos(
    ticker: str,
    periodo: str = Query(default="2y"),
    rc: RiskCalculator = Depends(get_risk_calculator),
):
    """
    Rendimientos logarítmicos y simples con estadísticas completas.
    Incluye pruebas de normalidad Jarque-Bera y Shapiro-Wilk.
    """
    ticker = ticker.upper()
    if ticker not in TICKERS_DEFAULT:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no disponible")

    try:
        return rc.estadisticas_rendimientos(ticker, periodo)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# ────────────────────────────────────────────────────────
# ENDPOINT 4: /indicadores/{ticker}
# ────────────────────────────────────────────────────────
@app.get("/indicadores/{ticker}", tags=["Análisis Técnico"])
async def indicadores_tecnicos(
    ticker: str,
    periodo: str = Query(default="1y"),
    ti: TechnicalIndicators = Depends(get_technical_indicators),
):
    """
    SMA, EMA, RSI, MACD, Bandas de Bollinger y Estocástico.
    Retorna series temporales completas para graficar en el frontend.
    """
    ticker = ticker.upper()
    if ticker not in TICKERS_DEFAULT:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no disponible")

    try:
        datos = ti.calcular_todos(ticker, periodo)
        return {"ticker": ticker, "periodo": periodo, "indicadores": datos}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# ────────────────────────────────────────────────────────
# ENDPOINT 5: /var — VaR y CVaR del portafolio
# ────────────────────────────────────────────────────────
@app.post("/var", tags=["Riesgo"])
async def calcular_var(
    request: PortafolioRequest,
    rc: RiskCalculator = Depends(get_risk_calculator),
):
    """
    Calcula VaR y CVaR con 3 métodos:
    - Paramétrico (distribución normal)
    - Simulación histórica
    - Montecarlo (10,000 simulaciones)
    """
    try:
        return rc.var_cvar(
            tickers=request.tickers,
            pesos=request.pesos,
            confianza=request.confianza,
            periodo=request.periodo,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# ────────────────────────────────────────────────────────
# ENDPOINT 6: /capm — Beta y retorno esperado CAPM
# ────────────────────────────────────────────────────────
@app.get("/capm", tags=["Análisis"])
async def capm_todos(
    periodo: str = Query(default="2y"),
    benchmark: str = Query(default="SPY"),
    rc: RiskCalculator = Depends(get_risk_calculator),
    macro: dict = Depends(get_macro_data),
):
    """
    Beta y retorno esperado CAPM para todos los activos del portafolio.
    La tasa libre de riesgo se obtiene automáticamente desde FRED API.
    Incluye clasificación: Agresivo (β>1.1), Defensivo (β<0.9), Neutro.
    """
    rf = macro.get("tasa_libre_riesgo", 5.25) / 100  # Convertir de % a decimal

    resultados = []
    for ticker in TICKERS_DEFAULT:
        try:
            resultado = rc.capm(ticker, benchmark, periodo, rf)
            resultados.append(resultado)
        except Exception as e:
            resultados.append({"ticker": ticker, "error": str(e)})

    return {
        "tasa_libre_riesgo_usada": rf,
        "benchmark": benchmark,
        "periodo": periodo,
        "resultados": resultados,
    }


# ────────────────────────────────────────────────────────
# ENDPOINT 7: /frontera-eficiente — Markowitz
# ────────────────────────────────────────────────────────
@app.post("/frontera-eficiente", tags=["Optimización"])
async def frontera_eficiente(
    request: PortafolioRequest,
    pa: PortfolioAnalyzer = Depends(get_portfolio_analyzer),
    macro: dict = Depends(get_macro_data),
):
    """
    Frontera eficiente de Markowitz con 10,000 portafolios simulados.
    Identifica el portafolio de máximo Sharpe y mínima varianza.
    Incluye heatmap de correlaciones.
    """
    try:
        rf = macro.get("tasa_libre_riesgo", 5.25) / 100
        return pa.frontera_eficiente(
            tickers=request.tickers,
            periodo=request.periodo,
            rf=rf,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# ────────────────────────────────────────────────────────
# ENDPOINT 8: /alertas — Señales de compra/venta
# ────────────────────────────────────────────────────────
@app.get("/alertas", tags=["Señales"])
async def alertas(
    pa: PortfolioAnalyzer = Depends(get_portfolio_analyzer),
):
    """
    Panel de señales automáticas para cada activo.
    Evalúa: RSI, cruce MACD, Bandas de Bollinger y Estocástico.
    Clasifica: COMPRA / VENTA / NEUTRAL con fuerza FUERTE / MODERADA / DÉBIL.
    """
    try:
        señales = pa.señales_alertas(TICKERS_DEFAULT)
        return {"alertas": señales, "total_activos": len(señales)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# ────────────────────────────────────────────────────────
# ENDPOINT 9: /macro — Indicadores macroeconómicos
# ────────────────────────────────────────────────────────
@app.get("/macro", tags=["Macro"])
async def macro_indicadores(
    macro: dict = Depends(get_macro_data),
):
    """
    Indicadores macroeconómicos en tiempo real desde FRED API.
    Incluye: tasa libre de riesgo (Fed Funds), inflación CPI, TRM USD/COP.
    """
    return macro


# ────────────────────────────────────────────────────────
# ENDPOINT BONUS: /garch/{ticker} — Modelos ARCH/GARCH
# ────────────────────────────────────────────────────────
@app.get("/garch/{ticker}", tags=["Análisis"])
async def garch(
    ticker: str,
    periodo: str = Query(default="2y"),
    rc: RiskCalculator = Depends(get_risk_calculator),
):
    """
    Ajusta y compara 3 modelos de volatilidad:
    ARCH(1), GARCH(1,1) y EGARCH(1,1).
    Retorna AIC, BIC, coeficientes y pronóstico de volatilidad.
    """
    ticker = ticker.upper()
    if ticker not in TICKERS_DEFAULT:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no disponible")

    try:
        return {"ticker": ticker, "modelos": rc.garch_analysis(ticker, periodo)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# ────────────────────────────────────────────────────────
# ENDPOINT BONUS: /benchmark — Comparación vs SPY
# ────────────────────────────────────────────────────────
@app.post("/benchmark", tags=["Optimización"])
async def benchmark_comparacion(
    request: PortafolioRequest,
    pa: PortfolioAnalyzer = Depends(get_portfolio_analyzer),
):
    """
    Compara el portafolio contra el benchmark (SPY).
    Métricas: Alpha de Jensen, Tracking Error, Information Ratio, Drawdown.
    """
    try:
        return pa.benchmark_comparacion(
            tickers=request.tickers,
            pesos=request.pesos,
            periodo=request.periodo,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# ── Health check ──
@app.get("/", tags=["Sistema"])
async def root():
    return {
        "mensaje": "🏦 API de Riesgo Financiero - Activa",
        "docs": "/docs",
        "endpoints_disponibles": [
            "GET  /activos",
            "GET  /precios/{ticker}",
            "GET  /rendimientos/{ticker}",
            "GET  /indicadores/{ticker}",
            "POST /var",
            "GET  /capm",
            "POST /frontera-eficiente",
            "GET  /alertas",
            "GET  /macro",
            "GET  /garch/{ticker}",
            "POST /benchmark",
        ]
    }
