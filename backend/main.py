from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uvicorn

from models import PortafolioRequest, HealthCheck
from services.datos import descargar_precios, obtener_info_activo, ACTIVOS_INFO
from services.indicadores import calcular_todos_indicadores
from services.riesgo import calcular_rendimientos, calcular_var_cvar
from services.portafolio import calcular_capm, calcular_frontera_eficiente

app = FastAPI(
    title="API de Análisis de Riesgo Financiero",
    description="""
Sistema de análisis de riesgo para portafolios de inversión.

**Módulos implementados:**
- Precios históricos reales (Yahoo Finance) ✅
- Indicadores técnicos: SMA, EMA, RSI, MACD, Bollinger ✅
- Rendimientos logarítmicos y pruebas de normalidad ✅
- Valor en Riesgo (VaR) y CVaR — 3 métodos ✅
- CAPM y Beta ✅
- Frontera eficiente de Markowitz ✅
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
    return HealthCheck(
        status="ok",
        mensaje="API de Riesgo Financiero funcionando correctamente",
        version="1.0.0",
        activos_disponibles=ACTIVOS,
    )


# ── ENDPOINT 2: LISTAR ACTIVOS ────────────────────────────────────────────────
@app.get("/activos", tags=["Activos"])
def listar_activos():
    activos = [{"ticker": t, **info} for t, info in ACTIVOS_INFO.items()]
    return {"total": len(activos), "activos": activos}


# ── ENDPOINT 2b: PRECIO ACTUAL ────────────────────────────────────────────────
@app.get("/activos/{ticker}/precio", tags=["Activos"])
def precio_actual(ticker: str):
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
    fecha_fin: Optional[str] = Query(default=None),
):
    """Rendimientos simples y logarítmicos con pruebas de normalidad."""
    ticker = ticker.upper()
    if ticker not in ACTIVOS:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no encontrado")
    try:
        return calcular_rendimientos(ticker, fecha_inicio, fecha_fin)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 5: INDICADORES TÉCNICOS ─────────────────────────────────────────
@app.get("/indicadores/{ticker}", tags=["Análisis"])
def obtener_indicadores(
    ticker: str,
    fecha_inicio: str = Query(default="2022-01-01"),
    fecha_fin: Optional[str] = Query(default=None),
):
    """SMA, EMA, RSI, MACD, Bollinger Bands, Estocástico + señales."""
    ticker = ticker.upper()
    if ticker not in ACTIVOS:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no encontrado")
    try:
        return calcular_todos_indicadores(ticker, fecha_inicio, fecha_fin)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 6: VaR y CVaR ────────────────────────────────────────────────────
@app.post("/var", tags=["Riesgo"])
def calcular_var(portafolio: PortafolioRequest):
    """VaR y CVaR con métodos histórico, paramétrico y Monte Carlo."""
    try:
        return calcular_var_cvar(
            tickers         = portafolio.tickers,
            pesos           = portafolio.pesos,
            fecha_inicio    = portafolio.fecha_inicio,
            fecha_fin       = portafolio.fecha_fin,
            nivel_confianza = portafolio.nivel_confianza,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 7: CAPM ──────────────────────────────────────────────────────────
@app.get("/capm", tags=["Riesgo"])
def obtener_capm(
    tickers: List[str] = Query(default=ACTIVOS),
    tasa_libre_riesgo: float = Query(default=0.0525, description="Tasa anual. Ej: 0.0525 = 5.25%"),
    fecha_inicio: str = Query(default="2022-01-01"),
    fecha_fin: Optional[str] = Query(default=None),
):
    """
    Calcula Beta y rendimiento esperado CAPM para cada activo.

    - **Beta > 1**: más volátil que el mercado
    - **Beta < 1**: menos volátil que el mercado
    - **Alpha > 0**: genera valor por encima de lo esperado
    - **R²**: qué tanto explica el mercado la variación del activo
    """
    tickers = [t.upper() for t in tickers]
    invalidos = [t for t in tickers if t not in ACTIVOS]
    if invalidos:
        raise HTTPException(status_code=404, detail=f"Tickers no válidos: {invalidos}. Disponibles: {ACTIVOS}")
    try:
        return calcular_capm(tickers, tasa_libre_riesgo=tasa_libre_riesgo,
                             fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 8: FRONTERA EFICIENTE ───────────────────────────────────────────
@app.post("/frontera-eficiente", tags=["Portafolio"])
def obtener_frontera(portafolio: PortafolioRequest):
    """
    Construye la frontera eficiente de Markowitz.

    Retorna:
    - **Portafolio de mínima varianza**: menor riesgo posible
    - **Portafolio de máximo Sharpe**: mejor balance riesgo/retorno
    - **Portafolio igual ponderado**: benchmark simple
    - **Frontera eficiente**: 50 puntos óptimos
    - **Simulación**: 500 portafolios aleatorios para graficar
    """
    try:
        return calcular_frontera_eficiente(
            tickers      = portafolio.tickers,
            fecha_inicio = portafolio.fecha_inicio,
            fecha_fin    = portafolio.fecha_fin,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 9: ALERTAS ───────────────────────────────────────────────────────
@app.get("/alertas", tags=["Señales"])
def obtener_alertas():
    """Señales de trading. Se implementará en la Fase 7."""
    return {"total_alertas": 0, "alertas": [], "nota": "Señales reales en Fase 7"}


# ── ENDPOINT 10: DATOS MACRO ──────────────────────────────────────────────────
@app.get("/macro", tags=["Macro"])
def obtener_macro():
    """Datos macroeconómicos FRED. Se implementará en la Fase 7."""
    return {"fuente": "FRED", "nota": "Conexión real en Fase 7",
            "datos_ejemplo": {"tasa_libre_riesgo": 0.0525}}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
