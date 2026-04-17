from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uvicorn

from models import PortafolioRequest, HealthCheck
from services.datos import descargar_precios, obtener_info_activo, ACTIVOS_INFO
from services.indicadores import calcular_todos_indicadores

app = FastAPI(
    title="API de Análisis de Riesgo Financiero",
    description="""
Sistema de análisis de riesgo para portafolios de inversión.

**Módulos implementados:**
- Precios históricos reales (Yahoo Finance) ✅
- Indicadores técnicos: SMA, EMA, RSI, MACD, Bollinger ✅
- Rendimientos y estadísticas
- Valor en Riesgo (VaR) y CVaR
- CAPM y Beta
- Frontera eficiente de Markowitz
- Señales de trading automatizadas
- Datos macroeconómicos (FRED)
    """,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ACTIVOS = list(ACTIVOS_INFO.keys())


# ── ENDPOINT 1: HEALTH CHECK ──────────────────────────────────────────────────
@app.get("/", response_model=HealthCheck, tags=["Sistema"])
def health_check():
    """Verifica que el servidor esté corriendo. Responde instantáneo."""
    return HealthCheck(
        status="ok",
        mensaje="API de Riesgo Financiero funcionando correctamente",
        version="1.0.0",
        activos_disponibles=ACTIVOS,
    )


# ── ENDPOINT 2: LISTAR ACTIVOS ────────────────────────────────────────────────
@app.get("/activos", tags=["Activos"])
def listar_activos():
    """Lista los activos del portafolio. Responde instantáneo."""
    activos = [
        {"ticker": t, **info, "nota": "Usa /precios/{ticker} para precios en tiempo real"}
        for t, info in ACTIVOS_INFO.items()
    ]
    return {"total": len(activos), "activos": activos}


# ── ENDPOINT 2b: PRECIO ACTUAL ────────────────────────────────────────────────
@app.get("/activos/{ticker}/precio", tags=["Activos"])
def precio_actual(ticker: str):
    """Precio actual de un activo desde Yahoo Finance."""
    ticker = ticker.upper()
    if ticker not in ACTIVOS:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no encontrado. Disponibles: {ACTIVOS}")
    return obtener_info_activo(ticker)


# ── ENDPOINT 3: PRECIOS HISTÓRICOS ───────────────────────────────────────────
@app.get("/precios/{ticker}", tags=["Precios"])
def obtener_precios(
    ticker: str,
    fecha_inicio: str = Query(default="2022-01-01", description="Formato YYYY-MM-DD"),
    fecha_fin: Optional[str] = Query(default=None, description="Formato YYYY-MM-DD"),
):
    """Precios históricos reales desde Yahoo Finance."""
    ticker = ticker.upper()
    if ticker not in ACTIVOS:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no encontrado. Disponibles: {ACTIVOS}")
    try:
        df = descargar_precios(ticker, fecha_inicio, fecha_fin)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {
        "ticker": ticker,
        "fecha_inicio": fecha_inicio,
        "fecha_fin": fecha_fin or "hoy",
        "total_dias": len(df),
        "fuente": "Yahoo Finance",
        "datos": df.to_dict(orient="records"),
    }


# ── ENDPOINT 4: RENDIMIENTOS ──────────────────────────────────────────────────
@app.get("/rendimientos/{ticker}", tags=["Análisis"])
def obtener_rendimientos(
    ticker: str,
    fecha_inicio: str = Query(default="2022-01-01"),
):
    """Rendimientos logarítmicos. Se implementará en la Fase 5."""
    ticker = ticker.upper()
    if ticker not in ACTIVOS:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no encontrado")
    return {"ticker": ticker, "nota": "Cálculo real en Fase 5"}


# ── ENDPOINT 5: INDICADORES TÉCNICOS ─────────────────────────────────────────
@app.get("/indicadores/{ticker}", tags=["Análisis"])
def obtener_indicadores(
    ticker: str,
    fecha_inicio: str = Query(default="2022-01-01", description="Formato YYYY-MM-DD"),
    fecha_fin: Optional[str] = Query(default=None, description="Formato YYYY-MM-DD"),
):
    """
    Calcula indicadores técnicos reales usando precios de Yahoo Finance.

    Indicadores incluidos:
    - **SMA 20, 50, 200** — Medias móviles simples
    - **EMA 20, 50** — Medias móviles exponenciales
    - **RSI 14** — Índice de fuerza relativa (0-100)
    - **MACD** — Convergencia/divergencia de medias móviles
    - **Bollinger Bands** — Bandas de volatilidad
    - **Estocástico** — Oscilador de momentum

    También retorna señales de compra/venta del último día.
    """
    ticker = ticker.upper()
    if ticker not in ACTIVOS:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no encontrado. Disponibles: {ACTIVOS}")
    try:
        resultado = calcular_todos_indicadores(ticker, fecha_inicio, fecha_fin)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculando indicadores: {e}")
    return resultado


# ── ENDPOINT 6: VaR ───────────────────────────────────────────────────────────
@app.post("/var", tags=["Riesgo"])
def calcular_var(portafolio: PortafolioRequest):
    """VaR y CVaR del portafolio. Se implementará en la Fase 5."""
    return {
        "tickers": portafolio.tickers,
        "pesos": portafolio.pesos,
        "nivel_confianza": portafolio.nivel_confianza,
        "nota": "Cálculo real en Fase 5",
        "resultado_ejemplo": {
            "var_historico": -0.0234,
            "cvar": -0.0312,
            "interpretacion": f"Con {portafolio.nivel_confianza*100:.0f}% de confianza, la pérdida máxima diaria no excedería el 2.34%",
        },
    }


# ── ENDPOINT 7: CAPM ──────────────────────────────────────────────────────────
@app.get("/capm", tags=["Riesgo"])
def calcular_capm(tickers: List[str] = Query(default=ACTIVOS)):
    """Beta y CAPM por activo. Se implementará en la Fase 6."""
    return {"benchmark": "S&P 500 (^GSPC)", "nota": "Cálculo real en Fase 6"}


# ── ENDPOINT 8: FRONTERA EFICIENTE ───────────────────────────────────────────
@app.post("/frontera-eficiente", tags=["Portafolio"])
def calcular_frontera(portafolio: PortafolioRequest):
    """Frontera eficiente de Markowitz. Se implementará en la Fase 6."""
    return {"tickers": portafolio.tickers, "nota": "Frontera eficiente en Fase 6"}


# ── ENDPOINT 9: ALERTAS ───────────────────────────────────────────────────────
@app.get("/alertas", tags=["Señales"])
def obtener_alertas():
    """Señales de trading. Se implementará en la Fase 7."""
    return {"total_alertas": 0, "alertas": [], "nota": "Señales reales en Fase 7"}


# ── ENDPOINT 10: DATOS MACRO ──────────────────────────────────────────────────
@app.get("/macro", tags=["Macro"])
def obtener_macro():
    """Datos macroeconómicos FRED. Se implementará en la Fase 7."""
    return {"fuente": "FRED", "nota": "Conexión real en Fase 7", "datos_ejemplo": {"tasa_libre_riesgo": 0.0525}}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
