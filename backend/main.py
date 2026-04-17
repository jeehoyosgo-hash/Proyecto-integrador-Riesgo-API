from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uvicorn
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
load_dotenv()  # también busca en la carpeta actual

from models import PortafolioRequest, HealthCheck
from services.datos import (
    descargar_precios, obtener_info_activo,
    CATALOGO, ACTIVOS_INFO,
    get_regiones, get_sectores, get_paises,
    get_por_region, get_por_sector, get_por_pais,
    BENCHMARKS,
)
from services.indicadores import calcular_todos_indicadores
from services.riesgo import calcular_rendimientos, calcular_var_cvar
from services.portafolio import calcular_capm, calcular_frontera_eficiente
from services.macro import generar_alertas_portafolio, obtener_datos_fred
from services.comparacion import comparar_activos, recomendar_portafolio

app = FastAPI(
    title="API de Análisis de Riesgo Financiero",
    description="""
Sistema de análisis de riesgo para portafolios de inversión — 30 activos globales.

**Regiones:** Norteamérica · Europa · América Latina · Asia
**Sectores:** Tecnología · Financiero · Energía · Salud · Consumo · Automotriz

**Módulos:**
- Precios históricos reales (Yahoo Finance) ✅
- Indicadores técnicos: SMA, EMA, RSI, MACD, Bollinger ✅
- Rendimientos logarítmicos y pruebas de normalidad ✅
- Valor en Riesgo (VaR) y CVaR — 3 métodos + Kupiec ✅
- CAPM, Beta y Alpha ✅
- Frontera eficiente de Markowitz ✅
- Señales automáticas de trading ✅
- Datos macroeconómicos FRED ✅
- Comparación de activos entre regiones/sectores ✅
- Motor de recomendaciones con scoring multifactor ✅
    """,
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TODOS_TICKERS = list(CATALOGO.keys())
ACTIVOS_BASE  = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
FRED_API_KEY  = os.getenv("FRED_API_KEY")


# ── ENDPOINT 1: HEALTH CHECK ──────────────────────────────────────────────────
@app.get("/", response_model=HealthCheck, tags=["Sistema"])
def health_check():
    return HealthCheck(
        status="ok",
        mensaje=f"API de Riesgo Financiero v2 — {len(CATALOGO)} activos globales",
        version="2.0.0",
        activos_disponibles=ACTIVOS_BASE,
    )


# ── ENDPOINT 2: LISTAR ACTIVOS ────────────────────────────────────────────────
@app.get("/activos", tags=["Activos"])
def listar_activos(
    region: Optional[str] = Query(default=None, description="Filtrar por región"),
    sector: Optional[str] = Query(default=None, description="Filtrar por sector"),
    pais:   Optional[str] = Query(default=None, description="Filtrar por país"),
):
    """
    Lista activos del portafolio global.
    Puedes filtrar por región, sector o país.
    """
    activos = []
    for ticker, info in CATALOGO.items():
        if region and info["region"] != region: continue
        if sector and info["sector"] != sector: continue
        if pais   and info["pais"]   != pais:   continue
        activos.append({"ticker": ticker, **info})

    return {
        "total":    len(activos),
        "activos":  activos,
        "regiones": get_regiones(),
        "sectores": get_sectores(),
        "paises":   get_paises(),
    }


# ── ENDPOINT 3: CATÁLOGO ──────────────────────────────────────────────────────
@app.get("/catalogo", tags=["Activos"])
def obtener_catalogo():
    """Retorna el catálogo completo organizado por región y sector."""
    por_region = {}
    por_sector = {}

    for ticker, info in CATALOGO.items():
        r = info["region"]
        s = info["sector"]
        if r not in por_region: por_region[r] = []
        if s not in por_sector: por_sector[s] = []
        entry = {"ticker": ticker, **info}
        por_region[r].append(entry)
        por_sector[s].append(entry)

    return {
        "total_activos": len(CATALOGO),
        "regiones":      get_regiones(),
        "sectores":      get_sectores(),
        "paises":        get_paises(),
        "por_region":    por_region,
        "por_sector":    por_sector,
        "benchmarks":    BENCHMARKS,
    }


# ── ENDPOINT 4: PRECIO ACTUAL ─────────────────────────────────────────────────
@app.get("/activos/{ticker}/precio", tags=["Activos"])
def precio_actual(ticker: str):
    ticker = ticker.upper()
    if ticker not in CATALOGO:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no encontrado.")
    return obtener_info_activo(ticker)


# ── ENDPOINT 5: PRECIOS HISTÓRICOS ───────────────────────────────────────────
@app.get("/precios/{ticker}", tags=["Precios"])
def obtener_precios(
    ticker: str,
    fecha_inicio: str = Query(default="2022-01-01"),
    fecha_fin: Optional[str] = Query(default=None),
):
    """Precios históricos reales desde Yahoo Finance. Soporta los 30 activos globales."""
    ticker = ticker.upper()
    if ticker not in CATALOGO:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no encontrado.")
    try:
        df = descargar_precios(ticker, fecha_inicio, fecha_fin)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))
    info = CATALOGO[ticker]
    return {
        "ticker": ticker, **info,
        "fecha_inicio": fecha_inicio,
        "fecha_fin":    fecha_fin or "hoy",
        "total_dias":   len(df),
        "fuente":       "Yahoo Finance",
        "datos":        df.to_dict(orient="records"),
    }


# ── ENDPOINT 6: RENDIMIENTOS ──────────────────────────────────────────────────
@app.get("/rendimientos/{ticker}", tags=["Análisis"])
def obtener_rendimientos(
    ticker: str,
    fecha_inicio: str = Query(default="2022-01-01"),
    fecha_fin: Optional[str] = Query(default=None),
):
    ticker = ticker.upper()
    if ticker not in CATALOGO:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no encontrado.")
    try:
        return calcular_rendimientos(ticker, fecha_inicio, fecha_fin)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 7: INDICADORES TÉCNICOS ─────────────────────────────────────────
@app.get("/indicadores/{ticker}", tags=["Análisis"])
def obtener_indicadores(
    ticker: str,
    fecha_inicio: str = Query(default="2022-01-01"),
    fecha_fin: Optional[str] = Query(default=None),
):
    ticker = ticker.upper()
    if ticker not in CATALOGO:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no encontrado.")
    try:
        return calcular_todos_indicadores(ticker, fecha_inicio, fecha_fin)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 8: VaR y CVaR ────────────────────────────────────────────────────
@app.post("/var", tags=["Riesgo"])
def calcular_var(portafolio: PortafolioRequest):
    """VaR y CVaR con métodos histórico, paramétrico y Monte Carlo + Kupiec."""
    invalidos = [t for t in portafolio.tickers if t not in CATALOGO]
    if invalidos:
        raise HTTPException(status_code=404, detail=f"Tickers no válidos: {invalidos}")
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


# ── ENDPOINT 9: CAPM ──────────────────────────────────────────────────────────
@app.get("/capm", tags=["Riesgo"])
def obtener_capm(
    tickers: List[str] = Query(default=ACTIVOS_BASE),
    tasa_libre_riesgo: float = Query(default=0.0525),
    fecha_inicio: str = Query(default="2022-01-01"),
    fecha_fin: Optional[str] = Query(default=None),
):
    tickers = [t.upper() for t in tickers]
    invalidos = [t for t in tickers if t not in CATALOGO]
    if invalidos:
        raise HTTPException(status_code=404, detail=f"Tickers no válidos: {invalidos}")
    try:
        return calcular_capm(tickers, tasa_libre_riesgo=tasa_libre_riesgo,
                             fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 10: FRONTERA EFICIENTE ───────────────────────────────────────────
@app.post("/frontera-eficiente", tags=["Portafolio"])
def obtener_frontera(portafolio: PortafolioRequest):
    invalidos = [t for t in portafolio.tickers if t not in CATALOGO]
    if invalidos:
        raise HTTPException(status_code=404, detail=f"Tickers no válidos: {invalidos}")
    try:
        return calcular_frontera_eficiente(
            tickers      = portafolio.tickers,
            fecha_inicio = portafolio.fecha_inicio,
            fecha_fin    = portafolio.fecha_fin,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 11: ALERTAS ──────────────────────────────────────────────────────
@app.get("/alertas", tags=["Señales"])
def obtener_alertas(
    tickers: List[str] = Query(default=ACTIVOS_BASE),
    fecha_inicio: str  = Query(default="2023-01-01"),
):
    tickers = [t.upper() for t in tickers]
    invalidos = [t for t in tickers if t not in CATALOGO]
    if invalidos:
        raise HTTPException(status_code=404, detail=f"Tickers no válidos: {invalidos}")
    try:
        return generar_alertas_portafolio(tickers, fecha_inicio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 12: DATOS MACRO ──────────────────────────────────────────────────
@app.get("/macro", tags=["Macro"])
def obtener_macro(
    series: List[str] = Query(
        default=["DGS3MO","DGS10","CPIAUCSL","UNRATE","FEDFUNDS","VIXCLS"]
    ),
):
    try:
        return obtener_datos_fred(api_key=FRED_API_KEY, series=series)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 13: COMPARAR ACTIVOS ─────────────────────────────────────────────
@app.get("/comparar", tags=["Comparación"])
def comparar(
    tickers: List[str] = Query(
        default=["AAPL","SAP.DE","TM","EC"],
        description="Lista de tickers a comparar (de cualquier región o sector)"
    ),
    fecha_inicio: str = Query(default="2022-01-01"),
    fecha_fin: Optional[str] = Query(default=None),
):
    """
    Compara múltiples activos de distintos países y sectores lado a lado.

    Métricas comparadas:
    - Retorno total y anualizado
    - Volatilidad anual
    - Sharpe Ratio
    - Máximo Drawdown
    - Momentum 3 meses
    - RSI actual y tendencia EMA

    Incluye ranking por Sharpe Ratio.
    """
    tickers = [t.upper() for t in tickers]
    invalidos = [t for t in tickers if t not in CATALOGO]
    if invalidos:
        raise HTTPException(status_code=404,
            detail=f"Tickers no válidos: {invalidos}. Usa GET /catalogo para ver todos.")
    if len(tickers) < 2:
        raise HTTPException(status_code=400, detail="Se necesitan al menos 2 tickers para comparar.")
    try:
        return comparar_activos(tickers, fecha_inicio, fecha_fin)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 14: RECOMENDAR PORTAFOLIO ────────────────────────────────────────
@app.get("/recomendar", tags=["Recomendaciones"])
def recomendar(
    perfil_riesgo: str = Query(
        default="moderado",
        description="Perfil de riesgo: conservador, moderado o agresivo"
    ),
    region: Optional[str] = Query(
        default=None,
        description="Limitar recomendación a una región específica"
    ),
    sector: Optional[str] = Query(
        default=None,
        description="Limitar recomendación a un sector específico"
    ),
    fecha_inicio: str = Query(default="2022-01-01"),
):
    """
    Motor de recomendaciones con scoring multifactor.

    **Perfiles disponibles:**
    - `conservador` — baja volatilidad, activos defensivos (Salud, Consumo)
    - `moderado`    — balance riesgo/retorno (por defecto)
    - `agresivo`    — maximiza retorno esperado (mayor concentración en Tecnología)

    **Scoring ponderado:**
    - Sharpe Ratio: 40% (moderado) / 30% (conservador) / 35% (agresivo)
    - Señales técnicas: 25%
    - Momentum 3M: 20% (moderado) / 10% (conservador) / 35% (agresivo)
    - Baja volatilidad: 15% (moderado) / 40% (conservador) / 5% (agresivo)

    Retorna portafolio sugerido con pesos, justificación y alertas de riesgo.
    """
    if perfil_riesgo not in ["conservador", "moderado", "agresivo"]:
        raise HTTPException(status_code=400,
            detail="perfil_riesgo debe ser: conservador, moderado o agresivo")

    # Construir universo de activos según filtros
    if region:
        tickers = get_por_region(region)
    elif sector:
        tickers = get_por_sector(sector)
    else:
        # Universo diversificado por defecto
        tickers = [
            "AAPL","MSFT","JPM","XOM","JNJ","AMZN",
            "EC","CIB","TM","INFY","NOVN.SW","HSBA.L",
            "SAP.DE","BAC","CVX","PFE","TSLA","SONY",
        ]

    if not tickers:
        raise HTTPException(status_code=404,
            detail=f"No se encontraron activos con los filtros indicados.")

    try:
        return recomendar_portafolio(
            tickers       = tickers,
            perfil_riesgo = perfil_riesgo,
            fecha_inicio  = fecha_inicio,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
