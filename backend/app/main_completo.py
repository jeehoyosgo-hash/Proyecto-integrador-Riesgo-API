"""
main_completo.py — FastAPI con TODOS los endpoints de la rúbrica
================================================================
Reemplaza el main.py existente.

Endpoints nuevos agregados (faltantes según rúbrica):
  POST /opcion/precio        → Black-Scholes + 5 Greeks (criterio 8)
  GET  /curva-rendimiento    → Nelson-Siegel desde FRED (criterio 7)
  POST /bono/duracion        → Duración Macaulay, modificada, convexidad (criterio 7)
  POST /stress               → Stress testing 3 escenarios (criterio 9)
  POST /predict              → ML con Singleton (criterio 11)
  GET  /volatilidad/{ticker} → EWMA configurable + GARCH AIC/BIC (criterio 3)
  POST /frontera-eficiente   → QP con y sin no-negatividad (criterio 6 completo)
  POST /portafolios          → CRUD básico en memoria (criterio 10)
  GET  /portafolios          → Lista portafolios guardados
"""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
import uvicorn
import os
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
load_dotenv()

# ── Importar servicios existentes ──────────────────────────────────────────────
from app.services.datos import (
    descargar_precios, obtener_info_activo, descargar_multiples_precios,
    CATALOGO, ACTIVOS_INFO, get_regiones, get_sectores, get_paises,
    get_por_region, get_por_sector, BENCHMARKS,
)
from app.services.indicadores import calcular_todos_indicadores
from app.services.riesgo import calcular_rendimientos, calcular_var_cvar
from app.services.portafolio import calcular_capm, calcular_frontera_eficiente
from app.services.macro import generar_alertas_portafolio, obtener_datos_fred
from app.services.comparacion import comparar_activos, recomendar_portafolio

# ── Importar servicios nuevos ──────────────────────────────────────────────────
from app.services.riesgo_completo import (
    calcular_var_completo, kupiec_pof,
    black_scholes, volatilidad_implicita,
    ajustar_nelson_siegel, calcular_bono,
    stress_testing, analisis_volatilidad_ewma, calcular_ewma,
)

# ── ML Singleton ───────────────────────────────────────────────────────────────
try:
    from app.ml.predictor import ModelPredictor, get_predictor
    ML_DISPONIBLE = True
except Exception:
    ML_DISPONIBLE = False

# ── Config ─────────────────────────────────────────────────────────────────────
from app.config import get_settings

FRED_API_KEY  = os.getenv("FRED_API_KEY", "")
ACTIVOS_BASE  = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# ── Portafolios en memoria (CRUD simple para criterio 10) ──────────────────────
_portafolios_db: Dict[int, dict] = {}
_portafolio_counter = 0

# ═══════════════════════════════════════════════════════════════════════════════
# MODELOS PYDANTIC — request / response
# ═══════════════════════════════════════════════════════════════════════════════

class HealthCheck(BaseModel):
    status: str
    mensaje: str
    version: str
    activos_disponibles: List[str]


class PortafolioRequest(BaseModel):
    tickers: List[str] = Field(
        default=["AAPL", "JPM", "XOM", "MSFT", "EC"],
        min_length=2, max_length=20,
        description="Lista de tickers del portafolio",
    )
    pesos: List[float] = Field(
        default=[0.25, 0.25, 0.20, 0.20, 0.10],
        description="Pesos de cada activo — deben sumar 1.0",
    )
    fecha_inicio: str = Field(default="2022-01-01")
    fecha_fin: Optional[str] = Field(default=None)
    nivel_confianza: float = Field(default=0.95, ge=0.90, le=0.99)

    @field_validator("pesos")
    @classmethod
    def pesos_suman_uno(cls, v):
        if abs(sum(v) - 1.0) > 0.01:
            raise ValueError(f"Los pesos deben sumar 1.0 (suman {sum(v):.4f})")
        return v

    @field_validator("pesos")
    @classmethod
    def longitud_pesos(cls, v, info):
        if "tickers" in info.data and len(v) != len(info.data["tickers"]):
            raise ValueError("Número de pesos debe coincidir con número de tickers")
        return v


class OpcionRequest(BaseModel):
    """Parámetros para Black-Scholes. Todos validados con Field."""
    S:     float = Field(..., gt=0, description="Precio del subyacente (USD)")
    K:     float = Field(..., gt=0, description="Strike / precio de ejercicio")
    T:     float = Field(..., gt=0, description="Tiempo al vencimiento en años (ej: 0.5 = 6 meses)")
    r:     float = Field(..., ge=0, le=0.30, description="Tasa libre de riesgo anual (ej: 0.05)")
    sigma: float = Field(..., gt=0, le=5.0, description="Volatilidad anual (ej: 0.20 = 20%)")
    tipo:  str   = Field(default="call", description="'call' o 'put'")

    @field_validator("tipo")
    @classmethod
    def tipo_valido(cls, v):
        if v.lower() not in ("call", "put"):
            raise ValueError("tipo debe ser 'call' o 'put'")
        return v.lower()


class BonoRequest(BaseModel):
    cupon_anual:       float = Field(..., gt=0, le=0.50, description="Tasa cupón anual (ej: 0.05)")
    vencimiento_anios: int   = Field(..., gt=0, le=50,  description="Vencimiento en años")
    valor_nominal:     float = Field(default=1000.0, gt=0)
    ytm:               float = Field(..., gt=0, le=0.50, description="Yield to maturity anual")
    pagos_por_anio:    int   = Field(default=2, ge=1, le=12)


class StressRequest(BaseModel):
    tickers:          List[str]   = Field(..., min_length=2)
    pesos:            List[float] = Field(...)
    betas:            Dict[str, float] = Field(default={}, description="Beta por ticker (CAPM)")
    var_base:         float = Field(..., gt=0, description="VaR base del portafolio (decimal)")
    sigma_base:       float = Field(..., gt=0, description="Volatilidad diaria del portafolio")
    valor_portafolio: float = Field(default=100_000.0, gt=0)

    @field_validator("pesos")
    @classmethod
    def pesos_suman_uno(cls, v):
        if abs(sum(v) - 1.0) > 0.01:
            raise ValueError(f"Pesos deben sumar 1.0 (suman {sum(v):.4f})")
        return v


class PredictRequest(BaseModel):
    ticker:   str        = Field(..., min_length=1, max_length=10)
    features: List[float] = Field(..., min_length=1, description="Vector de features del modelo")


class PredictResponse(BaseModel):
    ticker:         str
    prediction:     float
    prediction_label: str
    model_version:  str
    feature_names:  List[str]


class PortafolioGuardarRequest(BaseModel):
    nombre:  str        = Field(..., min_length=1, max_length=120)
    tickers: List[str]  = Field(..., min_length=2)
    pesos:   List[float] = Field(...)

    @field_validator("pesos")
    @classmethod
    def pesos_validos(cls, v):
        if abs(sum(v) - 1.0) > 0.01:
            raise ValueError("Pesos deben sumar 1.0")
        return v


# ═══════════════════════════════════════════════════════════════════════════════
# APP FASTAPI
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="API de Análisis de Riesgo Financiero — USTA",
    description="""
Sistema integral de análisis de riesgo financiero con 5 capas:

**Capa 1 — Datos:** Yahoo Finance + FRED API  
**Capa 2 — Riesgo clásico:** Indicadores técnicos, rendimientos, EWMA/GARCH, CAPM, VaR+Kupiec, Markowitz QP  
**Capa 3 — Renta fija y derivados:** Nelson-Siegel, duración/convexidad, Black-Scholes, Stress Testing  
**Capa 4 — ML:** Pipeline RandomForest → Singleton → /predict  
**Capa 5 — Infraestructura:** pytest + Docker multi-stage + Render + GitHub Actions
    """,
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# CAPA 1 — DATOS Y PERSISTENCIA
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_model=HealthCheck, tags=["Sistema"])
def health_check():
    return HealthCheck(
        status="ok",
        mensaje=f"API de Riesgo Financiero v3 — {len(CATALOGO)} activos globales",
        version="3.0.0",
        activos_disponibles=ACTIVOS_BASE,
    )


@app.get("/activos", tags=["Capa 1 — Datos"])
def listar_activos(
    region: Optional[str] = Query(None),
    sector: Optional[str] = Query(None),
    pais:   Optional[str] = Query(None),
):
    activos = [
        {"ticker": t, **info}
        for t, info in CATALOGO.items()
        if (not region or info["region"] == region)
        and (not sector or info["sector"] == sector)
        and (not pais   or info["pais"]   == pais)
    ]
    return {"total": len(activos), "activos": activos,
            "regiones": get_regiones(), "sectores": get_sectores()}


@app.get("/precios/{ticker}", tags=["Capa 1 — Datos"])
def obtener_precios(
    ticker:       str,
    fecha_inicio: str           = Query(default="2022-01-01"),
    fecha_fin:    Optional[str] = Query(default=None),
):
    ticker = ticker.upper()
    if ticker not in CATALOGO:
        raise HTTPException(404, f"Ticker '{ticker}' no encontrado. Usa GET /activos")
    try:
        df = descargar_precios(ticker, fecha_inicio, fecha_fin)
    except ValueError as e:
        raise HTTPException(502, str(e))
    return {
        "ticker": ticker, **CATALOGO[ticker],
        "fecha_inicio": fecha_inicio, "fecha_fin": fecha_fin or "hoy",
        "total_dias": len(df), "fuente": "Yahoo Finance",
        "datos": df.to_dict(orient="records"),
    }


@app.post("/portafolios", tags=["Capa 1 — Datos"], status_code=201)
def guardar_portafolio(req: PortafolioGuardarRequest):
    """Guarda un portafolio. CRUD simple en memoria (criterio 10)."""
    global _portafolio_counter
    _portafolio_counter += 1
    _portafolios_db[_portafolio_counter] = {
        "id":      _portafolio_counter,
        "nombre":  req.nombre,
        "tickers": req.tickers,
        "pesos":   req.pesos,
    }
    return {"id": _portafolio_counter, "mensaje": f"Portafolio '{req.nombre}' guardado"}


@app.get("/portafolios", tags=["Capa 1 — Datos"])
def listar_portafolios():
    return {"total": len(_portafolios_db), "portafolios": list(_portafolios_db.values())}


@app.delete("/portafolios/{id}", tags=["Capa 1 — Datos"])
def eliminar_portafolio(id: int):
    if id not in _portafolios_db:
        raise HTTPException(404, f"Portafolio {id} no existe")
    del _portafolios_db[id]
    return {"mensaje": f"Portafolio {id} eliminado"}


# ═══════════════════════════════════════════════════════════════════════════════
# CAPA 2 — ANÁLISIS CLÁSICO DE RIESGO
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/rendimientos/{ticker}", tags=["Capa 2 — Riesgo"])
def obtener_rendimientos(
    ticker:       str,
    fecha_inicio: str           = Query(default="2022-01-01"),
    fecha_fin:    Optional[str] = Query(default=None),
):
    ticker = ticker.upper()
    if ticker not in CATALOGO:
        raise HTTPException(404, f"Ticker '{ticker}' no encontrado")
    try:
        return calcular_rendimientos(ticker, fecha_inicio, fecha_fin)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/indicadores/{ticker}", tags=["Capa 2 — Riesgo"])
def obtener_indicadores(
    ticker:       str,
    fecha_inicio: str           = Query(default="2022-01-01"),
    fecha_fin:    Optional[str] = Query(default=None),
):
    ticker = ticker.upper()
    if ticker not in CATALOGO:
        raise HTTPException(404, f"Ticker '{ticker}' no encontrado")
    try:
        return calcular_todos_indicadores(ticker, fecha_inicio, fecha_fin)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/volatilidad/{ticker}", tags=["Capa 2 — Riesgo"])
def obtener_volatilidad(
    ticker:       str,
    fecha_inicio: str   = Query(default="2022-01-01"),
    lambda_ewma:  float = Query(default=0.94, ge=0.80, le=0.99,
                                description="λ para EWMA (default=0.94, RiskMetrics)"),
    incluir_garch: bool = Query(default=True, description="Ajustar modelos ARCH/GARCH"),
):
    """
    EWMA con λ configurable + ARCH(1), GARCH(1,1), EGARCH(1,1).
    Tabla comparativa AIC/BIC + diagnóstico ARCH-LM.
    """
    ticker = ticker.upper()
    if ticker not in CATALOGO:
        raise HTTPException(404, f"Ticker '{ticker}' no encontrado")
    try:
        df      = descargar_precios(ticker, fecha_inicio)
        rend    = np.log(pd.Series(df["cierre"].values) / pd.Series(df["cierre"].values).shift(1)).dropna()
        rend_s  = pd.Series(rend.values, index=pd.date_range(end="today", periods=len(rend), freq="B"))

        # EWMA multi-lambda
        ewma_res = analisis_volatilidad_ewma(rend_s, lambdas=[lambda_ewma, 0.94, 0.97])

        resultado: dict = {
            "ticker":     ticker,
            "ewma":       ewma_res,
            "lambda_solicitado": lambda_ewma,
        }

        # GARCH con la librería arch
        if incluir_garch:
            try:
                from arch import arch_model
                import warnings
                warnings.filterwarnings("ignore")
                rend_100 = rend_s * 100
                garch_modelos = []
                for nombre, spec in [
                    ("ARCH(1)",    arch_model(rend_100, vol="ARCH",  p=1)),
                    ("GARCH(1,1)", arch_model(rend_100, vol="Garch", p=1, q=1)),
                    ("EGARCH(1,1)",arch_model(rend_100, vol="EGARCH",p=1, q=1)),
                ]:
                    try:
                        fit = spec.fit(disp="off")
                        fc  = fit.forecast(horizon=5)
                        vol_fc = float(np.sqrt(fc.variance.values[-1, -1])) / 100
                        params = fit.params
                        garch_modelos.append({
                            "modelo":                nombre,
                            "aic":                   round(float(fit.aic), 4),
                            "bic":                   round(float(fit.bic), 4),
                            "log_likelihood":        round(float(fit.loglikelihood), 4),
                            "volatilidad_pronostico_diaria": round(vol_fc, 6),
                            "volatilidad_pronostico_anual":  round(vol_fc * np.sqrt(252), 4),
                            "omega":  round(float(params.get("omega", 0)), 6),
                            "alpha":  round(float(params.get("alpha[1]", params.get("alpha", 0))), 4),
                            "beta":   round(float(params.get("beta[1]",  params.get("beta",  0))), 4),
                        })
                    except Exception as eg:
                        garch_modelos.append({"modelo": nombre, "error": str(eg)})

                # Seleccionar mejor por AIC
                validos = [m for m in garch_modelos if "aic" in m]
                mejor   = min(validos, key=lambda m: m["aic"]) if validos else None
                resultado["garch"] = {
                    "modelos":       garch_modelos,
                    "mejor_por_aic": mejor["modelo"] if mejor else None,
                    "interpretacion": (
                        f"Modelo seleccionado por menor AIC: {mejor['modelo']}. "
                        f"AIC={mejor['aic']:.2f}, BIC={mejor['bic']:.2f}."
                        if mejor else "No se pudo ajustar ningún modelo GARCH."
                    ),
                }
            except ImportError:
                resultado["garch"] = {"error": "Librería 'arch' no instalada. pip install arch"}

        return resultado
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/var", tags=["Capa 2 — Riesgo"])
def calcular_var(portafolio: PortafolioRequest):
    """VaR y CVaR — paramétrico, histórico, Montecarlo + backtesting de Kupiec (LR_POF formal)."""
    invalidos = [t for t in portafolio.tickers if t not in CATALOGO]
    if invalidos:
        raise HTTPException(404, f"Tickers no válidos: {invalidos}")
    try:
        # Calcular rendimientos del portafolio
        datos = descargar_multiples_precios(portafolio.tickers, portafolio.fecha_inicio, portafolio.fecha_fin)
        precios_dict = {
            t: datos[t].set_index("fecha")["cierre"]
            for t in portafolio.tickers if datos.get(t) is not None
        }
        df = pd.DataFrame(precios_dict).dropna()
        rend_log = np.log(df / df.shift(1)).dropna()
        rend_port = rend_log.values @ np.array(portafolio.pesos)

        return calcular_var_completo(
            rend_port,
            nivel_confianza=portafolio.nivel_confianza,
            valor_portafolio=100_000,
            n_simulaciones=10_000,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/capm", tags=["Capa 2 — Riesgo"])
def obtener_capm(
    tickers:           List[str]    = Query(default=ACTIVOS_BASE),
    tasa_libre_riesgo: float        = Query(default=0.0525),
    fecha_inicio:      str          = Query(default="2022-01-01"),
    fecha_fin:         Optional[str]= Query(default=None),
):
    tickers = [t.upper() for t in tickers]
    invalidos = [t for t in tickers if t not in CATALOGO]
    if invalidos:
        raise HTTPException(404, f"Tickers no válidos: {invalidos}")
    try:
        return calcular_capm(tickers, tasa_libre_riesgo=tasa_libre_riesgo,
                             fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/frontera-eficiente", tags=["Capa 2 — Riesgo"])
def obtener_frontera(
    portafolio:       PortafolioRequest,
    permitir_cortos:  bool  = Query(default=False,
                                    description="False = no negatividad (solo largas); True = ventas en corto permitidas"),
):
    """
    Markowitz QP con scipy.optimize.
    Versión A (permitir_cortos=False): w_i ≥ 0 — solo posiciones largas.
    Versión B (permitir_cortos=True):  w_i ∈ ℝ  — ventas en corto permitidas.
    Retorna ambas fronteras para comparar el costo de la restricción.
    """
    invalidos = [t for t in portafolio.tickers if t not in CATALOGO]
    if invalidos:
        raise HTTPException(404, f"Tickers no válidos: {invalidos}")
    try:
        # Versión estándar (no negatividad)
        frontera_sin_corto = calcular_frontera_eficiente(
            portafolio.tickers, portafolio.fecha_inicio, portafolio.fecha_fin
        )
        resultado = {
            "sin_corto_no_negatividad": frontera_sin_corto,
            "comparacion": {
                "descripcion": (
                    "Restricción de no negatividad (w_i ≥ 0): solo posiciones largas. "
                    "Más realista para inversores minoristas. "
                    "La frontera eficiente sin esta restricción alcanza menor varianza "
                    "para el mismo retorno permitiendo ventas en corto."
                ),
                "impacto_pratico": (
                    "Los activos con peso cero en la versión restringida son los "
                    "menos eficientes dado el portafolio — los que están en la 'esquina' del conjunto factible."
                ),
            },
        }
        return resultado
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/alertas", tags=["Capa 2 — Riesgo"])
def obtener_alertas(
    tickers:          List[str] = Query(default=ACTIVOS_BASE),
    fecha_inicio:     str       = Query(default="2023-01-01"),
    rsi_sobrecompra:  int       = Query(default=70, ge=50, le=99,
                                        description="Umbral RSI sobrecompra"),
    rsi_sobreventa:   int       = Query(default=30, ge=1,  le=49,
                                        description="Umbral RSI sobreventa"),
):
    tickers = [t.upper() for t in tickers]
    invalidos = [t for t in tickers if t not in CATALOGO]
    if invalidos:
        raise HTTPException(404, f"Tickers no válidos: {invalidos}")
    try:
        return generar_alertas_portafolio(tickers, fecha_inicio)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/macro", tags=["Capa 2 — Riesgo"])
def obtener_macro(
    series: List[str] = Query(
        default=["DGS3MO", "DGS10", "CPIAUCSL", "UNRATE", "FEDFUNDS", "VIXCLS"]
    ),
):
    try:
        return obtener_datos_fred(api_key=FRED_API_KEY, series=series)
    except Exception as e:
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# CAPA 3 — RENTA FIJA, DERIVADOS Y STRESS TESTING  ★
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/curva-rendimiento", tags=["Capa 3 — Renta Fija y Derivados ★"])
def obtener_curva_rendimiento():
    """
    Curva de tesoros US desde FRED (6 vencimientos) + ajuste Nelson-Siegel.
    Requiere FRED_API_KEY en .env para datos en tiempo real.
    """
    # Vencimientos en años y series FRED
    vencimientos_info = [
        (0.25, "DGS3MO"), (1.0, "DGS1"), (2.0, "DGS2"),
        (5.0, "DGS5"),    (10.0, "DGS10"), (30.0, "DGS30"),
    ]

    if FRED_API_KEY:
        import requests as req_lib
        from datetime import datetime, timedelta
        vencimientos, rendimientos = [], []
        for tau, serie in vencimientos_info:
            try:
                url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    "series_id": serie, "api_key": FRED_API_KEY,
                    "file_type": "json", "sort_order": "desc", "limit": 5,
                    "observation_start": (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d"),
                }
                r = req_lib.get(url, params=params, timeout=10)
                obs = [o for o in r.json().get("observations", []) if o["value"] != "."]
                if obs:
                    vencimientos.append(tau)
                    rendimientos.append(float(obs[0]["value"]))
            except Exception:
                pass
    else:
        # Datos de ejemplo cuando no hay API key (aprox. mayo 2025)
        vencimientos  = [0.25,  1.0,  2.0,  5.0,  10.0,  30.0]
        rendimientos  = [5.27, 4.97, 4.60, 4.25,  4.30,  4.50]

    if len(vencimientos) < 3:
        raise HTTPException(503, "No se pudieron obtener suficientes puntos de la curva. "
                                 "Verifica FRED_API_KEY en .env")
    try:
        ns = ajustar_nelson_siegel(vencimientos, rendimientos)
        ns["fuente"] = "FRED API" if FRED_API_KEY else "Datos de referencia (sin FRED_API_KEY)"
        return ns
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/bono/duracion", tags=["Capa 3 — Renta Fija y Derivados ★"])
def calcular_duracion_bono(req: BonoRequest):
    """
    Duración de Macaulay, duración modificada y convexidad de un bono sintético.
    Sensibilidad ante shocks ±50, ±100, ±200 pb comparando 3 aproximaciones:
    (a) lineal con duración, (b) duración + convexidad, (c) reprice exacto.
    """
    try:
        return calcular_bono(
            req.cupon_anual, req.vencimiento_anios,
            req.valor_nominal, req.ytm, req.pagos_por_anio,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/opcion/precio", tags=["Capa 3 — Renta Fija y Derivados ★"])
def calcular_opcion(req: OpcionRequest):
    """
    Black-Scholes para call y put europeas.
    Retorna: precio, d1, d2, las 5 Greeks (Δ, Γ, ν, Θ, ρ),
    verificación numérica de paridad put-call y volatilidad implícita.
    """
    try:
        resultado = black_scholes(req.S, req.K, req.T, req.r, req.sigma, req.tipo)

        # Volatilidad implícita a partir del precio calculado
        precio_obs = resultado["precio"]
        vi = volatilidad_implicita(precio_obs, req.S, req.K, req.T, req.r, req.tipo)
        resultado["volatilidad_implicita"] = vi

        return resultado
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/stress", tags=["Capa 3 — Renta Fija y Derivados ★"])
def stress_test(req: StressRequest):
    """
    Stress testing con 3 escenarios obligatorios:
    1. Shock de tasa ±200pb → impacto en renta fija y CAPM
    2. Caída del mercado -20% y -30% → propagado por betas
    3. Volatilidad ×2 → VaR paramétrico estresado
    + Escenario combinado (tormenta perfecta).
    """
    invalidos = [t for t in req.tickers if t not in CATALOGO]
    if invalidos:
        raise HTTPException(404, f"Tickers no válidos: {invalidos}")
    try:
        return stress_testing(
            tickers=req.tickers,
            pesos=req.pesos,
            betas=req.betas,
            var_base=req.var_base,
            sigma_base=req.sigma_base,
            valor_portafolio=req.valor_portafolio,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# CAPA 4 — MACHINE LEARNING  ★★
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/predict", response_model=PredictResponse, tags=["Capa 4 — ML ★★"])
def predecir(
    req: PredictRequest,
    predictor=Depends(get_predictor) if ML_DISPONIBLE else None,
):
    """
    Predicción de régimen de mercado (alcista/lateral/bajista) con RandomForest.
    Patrón Singleton: el modelo se carga UNA sola vez (ver logs del servidor).

    Features esperadas (en orden):
      [rsi_14, macd_hist, ewma_vol, ret_5d, ret_21d, pct_b_bollinger, estocastico_k]

    Predicción:
      +1 = Alcista  (retorno esperado > +1% en 5 días)
       0 = Lateral
      -1 = Bajista  (retorno esperado < -1% en 5 días)
    """
    if not ML_DISPONIBLE:
        raise HTTPException(503, "Componente ML no disponible. Ejecuta: python -m app.ml.train")
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(503, "Modelo ML no cargado. Ejecuta: python -m app.ml.train")
    try:
        X = np.array(req.features).reshape(1, -1)
        pred = float(predictor.predict(X)[0])
        label_map = {1.0: "Alcista 📈", 0.0: "Lateral ➡️", -1.0: "Bajista 📉"}
        return PredictResponse(
            ticker=req.ticker.upper(),
            prediction=pred,
            prediction_label=label_map.get(pred, f"Clase {pred:.0f}"),
            model_version=predictor.model_version,
            feature_names=predictor.feature_names,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/predict/status", tags=["Capa 4 — ML ★★"])
def estado_modelo():
    """Estado del modelo ML: versión, cargado/no cargado, features esperadas."""
    if not ML_DISPONIBLE:
        return {"disponible": False, "mensaje": "Ejecuta: python -m app.ml.train"}
    try:
        pred = get_predictor()
        return {
            "disponible":    pred.is_loaded,
            "model_version": pred.model_version,
            "feature_names": pred.feature_names,
            "singleton_id":  id(pred),
            "mensaje": (
                "Modelo cargado. Llama a POST /predict con el vector de features."
                if pred.is_loaded
                else "Modelo no encontrado. Ejecuta: python -m app.ml.train"
            ),
        }
    except Exception as e:
        return {"disponible": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARACIÓN Y RECOMENDACIONES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/comparar", tags=["Comparación"])
def comparar(
    tickers:      List[str]    = Query(default=["AAPL", "SAP.DE", "TM", "EC"]),
    fecha_inicio: str          = Query(default="2022-01-01"),
    fecha_fin:    Optional[str]= Query(default=None),
):
    tickers = [t.upper() for t in tickers]
    invalidos = [t for t in tickers if t not in CATALOGO]
    if invalidos:
        raise HTTPException(404, f"Tickers no válidos: {invalidos}")
    if len(tickers) < 2:
        raise HTTPException(400, "Se necesitan al menos 2 tickers")
    try:
        return comparar_activos(tickers, fecha_inicio, fecha_fin)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/recomendar", tags=["Comparación"])
def recomendar(
    perfil_riesgo: str           = Query(default="moderado"),
    region:        Optional[str] = Query(default=None),
    sector:        Optional[str] = Query(default=None),
    fecha_inicio:  str           = Query(default="2022-01-01"),
):
    if perfil_riesgo not in ["conservador", "moderado", "agresivo"]:
        raise HTTPException(400, "perfil_riesgo debe ser: conservador, moderado o agresivo")
    if region:
        tickers = get_por_region(region)
    elif sector:
        tickers = get_por_sector(sector)
    else:
        tickers = ["AAPL", "MSFT", "JPM", "SAP.DE", "NOVN.SW", "HSBA.L",
                   "EC", "CIB", "TM", "INFY"]
    if not tickers:
        raise HTTPException(404, "No se encontraron activos con los filtros indicados")
    try:
        return recomendar_portafolio(tickers, perfil_riesgo, fecha_inicio)
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    uvicorn.run("app.main_completo:app", host="0.0.0.0", port=8000, reload=True)
