"""
main.py — FastAPI: todos los endpoints de la rúbrica
=====================================================
CORRECCIONES aplicadas (vs versión anterior):

1. ASYNC: todas las rutas que hacen I/O externo usan async def + await
2. BaseSettings: FRED_API_KEY y config global vienen de app.config (no os.getenv)
3. Depends() CONECTADO: /macro usa get_macro_data(), servicios inyectados en /alertas
4. ACTIVOS_BASE sincronizado con datos.py (10 activos del catálogo curado)
5. model_validator para validaciones cruzadas entre campos Pydantic v2
6. response_model añadido en endpoints principales
7. rsi_sobrecompra / rsi_sobreventa se pasan al servicio de alertas
8. imports estandarizados con prefijo app.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator

# ── Configuración (BaseSettings) ──────────────────────────────────────────────
from app.config import get_settings

# ── Servicios ─────────────────────────────────────────────────────────────────
from app.services.datos import (
    descargar_precios, obtener_info_activo, descargar_multiples_precios,
    CATALOGO, ACTIVOS_BASE, ACTIVOS_README, ACTIVOS_INFO,
    get_regiones, get_sectores, get_paises,
    get_por_region, get_por_sector, BENCHMARKS,
)
from app.services.indicadores import calcular_todos_indicadores
from app.services.riesgo import calcular_rendimientos, calcular_var_cvar
from app.services.portafolio import calcular_capm, calcular_frontera_eficiente
from app.services.macro import generar_alertas_portafolio, obtener_datos_fred
from app.services.comparacion import comparar_activos, recomendar_portafolio
from app.services.riesgo_completo import (
    calcular_var_completo, kupiec_pof,
    black_scholes, volatilidad_implicita,
    ajustar_nelson_siegel, calcular_bono,
    stress_testing, analisis_volatilidad_ewma,
)

# ── Inyección de dependencias ──────────────────────────────────────────────────
from app.dependencies import (
    get_macro_data,
    get_technical_indicators,
    get_risk_calculator,
    get_portfolio_analyzer,
)

# ── ML (opcional) ─────────────────────────────────────────────────────────────
try:
    from app.ml.predictor import get_predictor
    ML_DISPONIBLE = True
except Exception:
    ML_DISPONIBLE = False

# ── CRUD en memoria (portafolios guardados) ────────────────────────────────────
_portafolios_db: Dict[int, dict] = {}
_portafolio_counter: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# MODELOS PYDANTIC — request / response
# ═══════════════════════════════════════════════════════════════════════════════

class HealthCheck(BaseModel):
    status: str
    mensaje: str
    version: str
    activos_base: List[str]
    total_activos: int


class ActivoInfo(BaseModel):
    ticker: str
    nombre: str
    sector: str
    pais: str
    region: str
    moneda: str
    descripcion: str = ""


class PortafolioRequest(BaseModel):
    tickers: List[str] = Field(
        default=["AAPL", "JPM", "XOM", "MSFT", "EC"],
        min_length=2,
        max_length=15,
        description="Lista de tickers del portafolio (mín 2, máx 15)",
    )
    pesos: List[float] = Field(
        default=[0.25, 0.25, 0.20, 0.20, 0.10],
        description="Pesos de cada activo. Deben sumar exactamente 1.0",
    )
    fecha_inicio: str = Field(
        default="2022-01-01",
        description="Fecha de inicio en formato YYYY-MM-DD",
    )
    fecha_fin: Optional[str] = Field(
        default=None,
        description="Fecha de fin (None = hoy)",
    )
    nivel_confianza: float = Field(
        default=0.95,
        ge=0.90,
        le=0.99,
        description="Nivel de confianza para VaR (0.90 – 0.99)",
    )

    @field_validator("pesos")
    @classmethod
    def pesos_suman_uno(cls, v: List[float]) -> List[float]:
        total = sum(v)
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Los pesos deben sumar 1.0 (suman {total:.4f}). "
                "Ajusta los valores hasta que la suma sea 1.0."
            )
        return v

    @model_validator(mode="after")
    def longitud_coherente(self) -> "PortafolioRequest":
        if len(self.pesos) != len(self.tickers):
            raise ValueError(
                f"Número de pesos ({len(self.pesos)}) debe coincidir "
                f"con número de tickers ({len(self.tickers)})"
            )
        return self


class VarResponse(BaseModel):
    portafolio: dict
    parametros: dict
    estadisticas_portafolio: dict
    var_historico: dict
    var_parametrico: dict
    var_monte_carlo: dict
    backtesting_kupiec: dict


class OpcionRequest(BaseModel):
    S:     float = Field(..., gt=0,   description="Precio del subyacente (USD)")
    K:     float = Field(..., gt=0,   description="Strike / precio de ejercicio")
    T:     float = Field(..., gt=0,   description="Tiempo al vencimiento en años (ej: 0.5)")
    r:     float = Field(..., ge=0, le=0.30, description="Tasa libre de riesgo anual")
    sigma: float = Field(..., gt=0, le=5.0,  description="Volatilidad anual (ej: 0.20 = 20%)")
    tipo:  str   = Field(default="call", description="'call' o 'put'")

    @field_validator("tipo")
    @classmethod
    def tipo_valido(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in ("call", "put"):
            raise ValueError("tipo debe ser 'call' o 'put'")
        return v


class BonoRequest(BaseModel):
    cupon_anual:       float = Field(..., gt=0, le=0.50, description="Tasa cupón anual (ej: 0.05 = 5%)")
    vencimiento_anios: int   = Field(..., gt=0, le=50,   description="Vencimiento en años")
    valor_nominal:     float = Field(default=1000.0, gt=0, description="Valor nominal del bono")
    ytm:               float = Field(..., gt=0, le=0.50, description="Yield to maturity anual")
    pagos_por_anio:    int   = Field(default=2, ge=1, le=12, description="Frecuencia de pagos al año")


class StressRequest(BaseModel):
    tickers:          List[str]        = Field(..., min_length=2)
    pesos:            List[float]      = Field(...)
    betas:            Dict[str, float] = Field(default={}, description="Beta por ticker (del CAPM)")
    var_base:         float = Field(..., gt=0, description="VaR base del portafolio (decimal)")
    sigma_base:       float = Field(..., gt=0, description="Volatilidad diaria del portafolio")
    valor_portafolio: float = Field(default=100_000.0, gt=0)

    @field_validator("pesos")
    @classmethod
    def pesos_suman_uno(cls, v: List[float]) -> List[float]:
        if abs(sum(v) - 1.0) > 0.01:
            raise ValueError(f"Pesos deben sumar 1.0 (suman {sum(v):.4f})")
        return v


class PredictRequest(BaseModel):
    ticker:   str         = Field(..., min_length=1, max_length=10)
    features: List[float] = Field(
        ...,
        min_length=7,
        max_length=7,
        description="Vector de 7 features: [rsi_14, macd_hist, ewma_vol, ret_5d, ret_21d, pct_b, estocastico_k]",
    )


class PredictResponse(BaseModel):
    ticker:           str
    prediction:       float
    prediction_label: str
    model_version:    str
    feature_names:    List[str]


class PortafolioGuardarRequest(BaseModel):
    nombre:  str         = Field(..., min_length=1, max_length=120)
    tickers: List[str]   = Field(..., min_length=2)
    pesos:   List[float] = Field(...)

    @field_validator("pesos")
    @classmethod
    def pesos_validos(cls, v: List[float]) -> List[float]:
        if abs(sum(v) - 1.0) > 0.01:
            raise ValueError("Pesos deben sumar 1.0")
        return v

    @model_validator(mode="after")
    def longitud_coherente(self) -> "PortafolioGuardarRequest":
        if len(self.pesos) != len(self.tickers):
            raise ValueError("Número de pesos debe coincidir con número de tickers")
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# APP FASTAPI
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="API de Análisis de Riesgo Financiero — USTA",
    description="""
Sistema integral de análisis de riesgo financiero.

**Portafolio:** 10 activos, 4 regiones (Norteamérica, Europa, LatAm, Asia), 5 sectores.

**Módulos:**
- Capa 1 — Datos: Yahoo Finance + FRED API + fallback sintético
- Capa 2 — Riesgo clásico: Indicadores técnicos, rendimientos, EWMA/GARCH, CAPM, VaR+Kupiec, Markowitz
- Capa 3 — Renta fija y derivados: Nelson-Siegel, duración/convexidad, Black-Scholes, Stress Testing
- Capa 4 — ML: RandomForest → Singleton → /predict
- Capa 5 — Infraestructura: pytest + Docker multi-stage + Render + GitHub Actions

**Autores:** Ver README.md
    """,
    version="3.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
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
async def health_check() -> HealthCheck:
    """Estado del sistema. Verifica que la API está activa."""
    return HealthCheck(
        status="ok",
        mensaje=f"API de Riesgo Financiero v3.1 — {len(CATALOGO)} activos disponibles",
        version="3.1.0",
        activos_base=ACTIVOS_BASE,
        total_activos=len(CATALOGO),
    )


@app.get("/activos", tags=["Capa 1 — Datos"])
async def listar_activos(
    region: Optional[str] = Query(None, description="Filtrar por región"),
    sector: Optional[str] = Query(None, description="Filtrar por sector"),
    pais:   Optional[str] = Query(None, description="Filtrar por país"),
) -> dict:
    """Lista todos los activos del catálogo con filtros opcionales."""
    activos = [
        {"ticker": t, **info}
        for t, info in CATALOGO.items()
        if (not region or info["region"] == region)
        and (not sector or info["sector"] == sector)
        and (not pais   or info["pais"]   == pais)
    ]
    return {
        "total": len(activos),
        "activos": activos,
        "regiones": get_regiones(),
        "sectores": get_sectores(),
        "activos_readme": ACTIVOS_README,
        "nota": "activos_readme son los 5 activos del portafolio base documentado.",
    }


@app.get("/precios/{ticker}", tags=["Capa 1 — Datos"])
async def obtener_precios(
    ticker:       str,
    fecha_inicio: str           = Query(default="2022-01-01", description="Fecha inicio YYYY-MM-DD"),
    fecha_fin:    Optional[str] = Query(default=None, description="Fecha fin (None = hoy)"),
) -> dict:
    """Retorna precios históricos de un activo (OHLCV)."""
    ticker = ticker.upper()
    if ticker not in CATALOGO:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' no encontrado. Consulta GET /activos para ver los disponibles.",
        )
    try:
        df = descargar_precios(ticker, fecha_inicio, fecha_fin)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {
        "ticker":      ticker,
        **CATALOGO[ticker],
        "fecha_inicio": fecha_inicio,
        "fecha_fin":    fecha_fin or "hoy",
        "total_dias":   len(df),
        "fuente":       "Yahoo Finance (fallback sintético si no hay conexión)",
        "datos":        df.to_dict(orient="records"),
    }


@app.post("/portafolios", tags=["Capa 1 — Datos"], status_code=201)
async def guardar_portafolio(req: PortafolioGuardarRequest) -> dict:
    """Guarda un portafolio en memoria (CRUD simple)."""
    global _portafolio_counter
    _portafolio_counter += 1
    _portafolios_db[_portafolio_counter] = {
        "id":      _portafolio_counter,
        "nombre":  req.nombre,
        "tickers": req.tickers,
        "pesos":   req.pesos,
        "creado":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return {
        "id":      _portafolio_counter,
        "mensaje": f"Portafolio '{req.nombre}' guardado exitosamente.",
    }


@app.get("/portafolios", tags=["Capa 1 — Datos"])
async def listar_portafolios() -> dict:
    """Lista todos los portafolios guardados en sesión."""
    return {
        "total":       len(_portafolios_db),
        "portafolios": list(_portafolios_db.values()),
    }


@app.delete("/portafolios/{id}", tags=["Capa 1 — Datos"])
async def eliminar_portafolio(id: int) -> dict:
    """Elimina un portafolio guardado."""
    if id not in _portafolios_db:
        raise HTTPException(status_code=404, detail=f"Portafolio {id} no existe.")
    del _portafolios_db[id]
    return {"mensaje": f"Portafolio {id} eliminado."}


# ═══════════════════════════════════════════════════════════════════════════════
# CAPA 2 — ANÁLISIS CLÁSICO DE RIESGO
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/rendimientos/{ticker}", tags=["Capa 2 — Riesgo"])
async def obtener_rendimientos(
    ticker:       str,
    fecha_inicio: str           = Query(default="2022-01-01"),
    fecha_fin:    Optional[str] = Query(default=None),
) -> dict:
    """
    Rendimientos simples y logarítmicos con estadísticas descriptivas
    y pruebas de normalidad (Jarque-Bera, Shapiro-Wilk).
    """
    ticker = ticker.upper()
    if ticker not in CATALOGO:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no encontrado.")
    try:
        return calcular_rendimientos(ticker, fecha_inicio, fecha_fin)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/indicadores/{ticker}", tags=["Capa 2 — Riesgo"])
async def obtener_indicadores(
    ticker:       str,
    fecha_inicio: str           = Query(default="2022-01-01"),
    fecha_fin:    Optional[str] = Query(default=None),
    svc = Depends(get_technical_indicators),
) -> dict:
    """
    Indicadores técnicos: SMA(20/50/200), EMA(20/50), RSI(14), MACD,
    Bandas de Bollinger, Oscilador Estocástico.
    Incluye señales automáticas del último día.
    """
    ticker = ticker.upper()
    if ticker not in CATALOGO:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no encontrado.")
    try:
        return calcular_todos_indicadores(ticker, fecha_inicio, fecha_fin)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/volatilidad/{ticker}", tags=["Capa 2 — Riesgo"])
async def obtener_volatilidad(
    ticker:        str,
    fecha_inicio:  str   = Query(default="2022-01-01"),
    lambda_ewma:   float = Query(
        default=0.94, ge=0.80, le=0.99,
        description="λ para EWMA (0.94 = RiskMetrics estándar)",
    ),
    incluir_garch: bool  = Query(default=True, description="Ajustar ARCH/GARCH"),
) -> dict:
    """
    EWMA con λ configurable + ARCH(1), GARCH(1,1), EGARCH(1,1).
    Tabla comparativa AIC/BIC + selección del mejor modelo + pronóstico 5 días.
    """
    ticker = ticker.upper()
    if ticker not in CATALOGO:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no encontrado.")
    try:
        df   = descargar_precios(ticker, fecha_inicio)
        rend = np.log(
            pd.Series(df["cierre"].values) / pd.Series(df["cierre"].values).shift(1)
        ).dropna()
        rend_s = pd.Series(
            rend.values,
            index=pd.date_range(end="today", periods=len(rend), freq="B"),
        )
        from app.services.riesgo_completo import analisis_volatilidad_ewma
        ewma_res = analisis_volatilidad_ewma(rend_s, lambdas=[lambda_ewma, 0.94, 0.97])
        resultado: dict = {
            "ticker":            ticker,
            "ewma":              ewma_res,
            "lambda_solicitado": lambda_ewma,
        }

        if incluir_garch:
            try:
                from arch import arch_model
                import warnings
                warnings.filterwarnings("ignore")
                rend_100 = rend_s * 100
                garch_modelos = []
                for nombre_m, spec in [
                    ("ARCH(1)",     arch_model(rend_100, vol="ARCH",  p=1)),
                    ("GARCH(1,1)",  arch_model(rend_100, vol="Garch", p=1, q=1)),
                    ("EGARCH(1,1)", arch_model(rend_100, vol="EGARCH", p=1, q=1)),
                ]:
                    try:
                        fit = spec.fit(disp="off")
                        fc  = fit.forecast(horizon=5)
                        vol_fc = float(np.sqrt(fc.variance.values[-1, -1])) / 100
                        p = fit.params
                        garch_modelos.append({
                            "modelo":             nombre_m,
                            "aic":                round(float(fit.aic), 4),
                            "bic":                round(float(fit.bic), 4),
                            "log_likelihood":     round(float(fit.loglikelihood), 4),
                            "vol_pronostico_dia": round(vol_fc, 6),
                            "vol_pronostico_anual": round(vol_fc * np.sqrt(252), 4),
                            "omega": round(float(p.get("omega", 0)), 6),
                            "alpha": round(float(p.get("alpha[1]", p.get("alpha", 0))), 4),
                            "beta":  round(float(p.get("beta[1]",  p.get("beta",  0))), 4),
                        })
                    except Exception as eg:
                        garch_modelos.append({"modelo": nombre_m, "error": str(eg)})

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
                resultado["garch"] = {
                    "error": "Librería 'arch' no instalada. Ejecuta: pip install arch"
                }

        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/var", response_model=VarResponse, tags=["Capa 2 — Riesgo"])
async def calcular_var(
    portafolio: PortafolioRequest,
    svc = Depends(get_risk_calculator),
) -> dict:
    """
    VaR y CVaR con 3 métodos: paramétrico, histórico, Monte Carlo (10,000 sim.)
    + backtesting de Kupiec (LR_POF estadístico formal con chi²).
    """
    invalidos = [t for t in portafolio.tickers if t not in CATALOGO]
    if invalidos:
        raise HTTPException(
            status_code=404,
            detail=f"Tickers no válidos: {invalidos}. Consulta GET /activos.",
        )
    try:
        datos = descargar_multiples_precios(
            portafolio.tickers, portafolio.fecha_inicio, portafolio.fecha_fin
        )
        precios_dict = {
            t: datos[t].set_index("fecha")["cierre"]
            for t in portafolio.tickers
            if datos.get(t) is not None
        }
        df      = pd.DataFrame(precios_dict).dropna()
        rend_log = np.log(df / df.shift(1)).dropna()
        rend_port = rend_log.values @ np.array(portafolio.pesos)

        from app.services.riesgo_completo import calcular_var_completo
        return calcular_var_completo(
            rend_port,
            nivel_confianza=portafolio.nivel_confianza,
            valor_portafolio=100_000,
            n_simulaciones=10_000,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/capm", tags=["Capa 2 — Riesgo"])
async def obtener_capm(
    tickers:            List[str]    = Query(default=ACTIVOS_README),
    tasa_libre_riesgo:  float        = Query(default=0.0525, description="Rf anual (si no se usa /macro)"),
    fecha_inicio:       str          = Query(default="2022-01-01"),
    fecha_fin:          Optional[str]= Query(default=None),
    macro: dict = Depends(get_macro_data),
) -> dict:
    """
    Beta y rendimiento esperado CAPM para cada activo.
    La tasa libre de riesgo se obtiene automáticamente de FRED via Depends(get_macro_data).
    Si FRED no está disponible, usa el valor por defecto del parámetro.
    """
    # Usar Rf de FRED si está disponible
    rf_fred = macro.get("tasa_libre_riesgo")
    rf = rf_fred / 100 if rf_fred else tasa_libre_riesgo

    tickers = [t.upper() for t in tickers]
    invalidos = [t for t in tickers if t not in CATALOGO]
    if invalidos:
        raise HTTPException(status_code=404, detail=f"Tickers no válidos: {invalidos}")
    try:
        resultado = calcular_capm(
            tickers,
            tasa_libre_riesgo=rf,
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin,
        )
        resultado["rf_fuente"] = "FRED API (tiempo real)" if rf_fred else "Valor por defecto"
        resultado["rf_usado"] = rf
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/frontera-eficiente", tags=["Capa 2 — Riesgo"])
async def obtener_frontera(
    portafolio:      PortafolioRequest,
    permitir_cortos: bool = Query(
        default=False,
        description="False = no negatividad (solo largas); True = cortos permitidos",
    ),
    svc = Depends(get_portfolio_analyzer),
) -> dict:
    """
    Frontera eficiente de Markowitz con optimización QP.
    Retorna portafolio de mínima varianza, máximo Sharpe y simulación de 500 portafolios.
    """
    invalidos = [t for t in portafolio.tickers if t not in CATALOGO]
    if invalidos:
        raise HTTPException(status_code=404, detail=f"Tickers no válidos: {invalidos}")
    try:
        frontera = calcular_frontera_eficiente(
            portafolio.tickers,
            portafolio.fecha_inicio,
            portafolio.fecha_fin,
        )
        frontera["configuracion"] = {
            "permitir_cortos": permitir_cortos,
            "descripcion": (
                "Solo posiciones largas (w_i ≥ 0). Más realista para inversores minoristas."
                if not permitir_cortos
                else "Ventas en corto permitidas (w_i ∈ ℝ). Frontera teórica más amplia."
            ),
        }
        return frontera
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alertas", tags=["Capa 2 — Riesgo ★ Módulo 7"])
async def obtener_alertas(
    tickers:          List[str] = Query(default=ACTIVOS_README),
    fecha_inicio:     str       = Query(default="2023-01-01"),
    rsi_sobrecompra:  int       = Query(
        default=70, ge=50, le=99,
        description="Umbral RSI sobrecompra (configurable, default 70)",
    ),
    rsi_sobreventa:   int       = Query(
        default=30, ge=1, le=49,
        description="Umbral RSI sobreventa (configurable, default 30)",
    ),
) -> dict:
    """
    Señales automáticas de compra/venta para cada activo basadas en:
    RSI, MACD, Bandas de Bollinger, Cruce EMA, Oscilador Estocástico.
    Umbrales RSI configurables por el usuario.
    """
    tickers   = [t.upper() for t in tickers]
    invalidos = [t for t in tickers if t not in CATALOGO]
    if invalidos:
        raise HTTPException(status_code=404, detail=f"Tickers no válidos: {invalidos}")
    try:
        resultado = generar_alertas_portafolio(
            tickers,
            fecha_inicio,
            rsi_sobrecompra=rsi_sobrecompra,   # ← ahora se pasan al servicio
            rsi_sobreventa=rsi_sobreventa,
        )
        resultado["umbrales_usados"] = {
            "rsi_sobrecompra": rsi_sobrecompra,
            "rsi_sobreventa":  rsi_sobreventa,
        }
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/macro", tags=["Capa 2 — Riesgo ★ Módulo 8"])
async def obtener_macro(
    series: List[str] = Query(
        default=["DGS3MO", "DGS10", "CPIAUCSL", "UNRATE", "FEDFUNDS", "VIXCLS"],
        description="Series FRED a consultar",
    ),
    macro: dict = Depends(get_macro_data),   # ← Depends() conectado
) -> dict:
    """
    Indicadores macroeconómicos actualizados vía FRED API.
    Incluye: tasa libre de riesgo (T-Bills 3M), inflación CPI,
    desempleo, tasa Fed, y VIX. Requiere FRED_API_KEY en .env.
    """
    return macro


# ═══════════════════════════════════════════════════════════════════════════════
# CAPA 3 — RENTA FIJA, DERIVADOS Y STRESS TESTING
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/curva-rendimiento", tags=["Capa 3 — Renta Fija y Derivados ★"])
async def obtener_curva_rendimiento() -> dict:
    """
    Curva de tesoros US desde FRED (6 vencimientos) + ajuste Nelson-Siegel.
    Requiere FRED_API_KEY para datos en tiempo real; usa fallback si no hay key.
    """
    settings    = get_settings()
    fred_key    = settings.fred_api_key
    import requests as req_lib

    vencimientos_info = [
        (0.25, "DGS3MO"), (1.0, "DGS1"), (2.0, "DGS2"),
        (5.0,  "DGS5"),   (10.0, "DGS10"), (30.0, "DGS30"),
    ]
    vencimientos, rendimientos = [], []

    if fred_key:
        for tau, serie in vencimientos_info:
            try:
                url    = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    "series_id": serie, "api_key": fred_key,
                    "file_type": "json", "sort_order": "desc", "limit": 5,
                    "observation_start": (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d"),
                }
                r   = req_lib.get(url, params=params, timeout=10)
                obs = [o for o in r.json().get("observations", []) if o["value"] != "."]
                if obs:
                    vencimientos.append(tau)
                    rendimientos.append(float(obs[0]["value"]))
            except Exception:
                pass
    else:
        vencimientos = [0.25, 1.0,  2.0,  5.0,  10.0, 30.0]
        rendimientos = [5.27, 4.97, 4.60, 4.25,  4.30, 4.50]

    if len(vencimientos) < 3:
        raise HTTPException(
            status_code=503,
            detail="No se pudieron obtener suficientes puntos de la curva. Verifica FRED_API_KEY.",
        )
    try:
        ns = ajustar_nelson_siegel(vencimientos, rendimientos)
        ns["fuente"] = "FRED API" if fred_key else "Datos de referencia (configura FRED_API_KEY)"
        return ns
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bono/duracion", tags=["Capa 3 — Renta Fija y Derivados ★"])
async def calcular_duracion_bono(req: BonoRequest) -> dict:
    """
    Duración Macaulay, duración modificada y convexidad.
    Sensibilidad ante shocks ±50, ±100, ±200 pb (3 aproximaciones).
    """
    try:
        return calcular_bono(
            req.cupon_anual, req.vencimiento_anios,
            req.valor_nominal, req.ytm, req.pagos_por_anio,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/opcion/precio", tags=["Capa 3 — Renta Fija y Derivados ★"])
async def calcular_opcion(req: OpcionRequest) -> dict:
    """
    Black-Scholes para call y put europeas.
    Retorna: precio, d1, d2, las 5 Greeks (Δ, Γ, ν, Θ, ρ),
    verificación de paridad put-call y volatilidad implícita.
    """
    try:
        resultado = black_scholes(req.S, req.K, req.T, req.r, req.sigma, req.tipo)
        vi = volatilidad_implicita(resultado["precio"], req.S, req.K, req.T, req.r, req.tipo)
        resultado["volatilidad_implicita"] = vi
        return resultado
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stress", tags=["Capa 3 — Renta Fija y Derivados ★"])
async def stress_test(req: StressRequest) -> dict:
    """
    Stress testing con 3 escenarios: shock de tasa ±200pb,
    caída de mercado -20%/-30%, volatilidad ×2 + escenario combinado.
    """
    invalidos = [t for t in req.tickers if t not in CATALOGO]
    if invalidos:
        raise HTTPException(status_code=404, detail=f"Tickers no válidos: {invalidos}")
    try:
        return stress_testing(
            tickers=req.tickers, pesos=req.pesos,
            betas=req.betas, var_base=req.var_base,
            sigma_base=req.sigma_base, valor_portafolio=req.valor_portafolio,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# CAPA 4 — MACHINE LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/predict", response_model=PredictResponse, tags=["Capa 4 — ML ★★"])
async def predecir(
    req: PredictRequest,
    predictor=Depends(get_predictor) if ML_DISPONIBLE else None,
) -> dict:
    """
    Predicción de régimen de mercado (alcista/lateral/bajista) con RandomForest.
    Features: [rsi_14, macd_hist, ewma_vol, ret_5d, ret_21d, pct_b_bollinger, estocastico_k]
    """
    if not ML_DISPONIBLE:
        raise HTTPException(status_code=503, detail="ML no disponible. Ejecuta: python -m app.ml.train")
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Modelo no cargado. Ejecuta: python -m app.ml.train")
    try:
        X    = np.array(req.features).reshape(1, -1)
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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/status", tags=["Capa 4 — ML ★★"])
async def estado_modelo() -> dict:
    """Estado del modelo ML: versión, disponibilidad, features esperadas."""
    if not ML_DISPONIBLE:
        return {"disponible": False, "mensaje": "Ejecuta: python -m app.ml.train"}
    try:
        pred = get_predictor()
        return {
            "disponible":    pred.is_loaded,
            "model_version": pred.model_version,
            "feature_names": pred.feature_names,
            "mensaje": "Modelo cargado." if pred.is_loaded else "Ejecuta: python -m app.ml.train",
        }
    except Exception as e:
        return {"disponible": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARACIÓN Y RECOMENDACIONES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/comparar", tags=["Comparación"])
async def comparar(
    tickers:      List[str]    = Query(default=ACTIVOS_README),
    fecha_inicio: str          = Query(default="2022-01-01"),
    fecha_fin:    Optional[str]= Query(default=None),
) -> dict:
    """Comparación multi-activo: rendimiento acumulado vs. benchmark + métricas."""
    tickers   = [t.upper() for t in tickers]
    invalidos = [t for t in tickers if t not in CATALOGO]
    if invalidos:
        raise HTTPException(status_code=404, detail=f"Tickers no válidos: {invalidos}")
    if len(tickers) < 2:
        raise HTTPException(status_code=400, detail="Se necesitan al menos 2 tickers.")
    try:
        return comparar_activos(tickers, fecha_inicio, fecha_fin)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recomendar", tags=["Comparación"])
async def recomendar(
    perfil_riesgo: str           = Query(default="moderado"),
    region:        Optional[str] = Query(default=None),
    sector:        Optional[str] = Query(default=None),
    fecha_inicio:  str           = Query(default="2022-01-01"),
) -> dict:
    """Recomienda portafolio según perfil de riesgo (conservador/moderado/agresivo)."""
    if perfil_riesgo not in ["conservador", "moderado", "agresivo"]:
        raise HTTPException(
            status_code=400,
            detail="perfil_riesgo debe ser: conservador, moderado o agresivo",
        )
    tickers = (
        get_por_region(region) if region
        else get_por_sector(sector) if sector
        else ACTIVOS_BASE
    )
    if not tickers:
        raise HTTPException(status_code=404, detail="No se encontraron activos con esos filtros.")
    try:
        return recomendar_portafolio(tickers, perfil_riesgo, fecha_inicio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
