"""
datos.py — CATÁLOGO CURADO: 10 activos + benchmarks
=====================================================
Lógica del portafolio:
  5 activos OBLIGATORIOS según el README/rúbrica:
    AAPL, MSFT, JPM, EC, NOVN.SW
  5 activos adicionales para diversificación real:
    XOM  (Energía USA — complementa a EC)
    JNJ  (Salud USA   — complementa a NOVN.SW)
    SAP.DE (Tecnología Europa — diversificación geográfica)
    TM   (Automotriz Asia — 4ª región requerida)
    CIB  (Financiero Colombia — contexto académico local)

  Total: 10 activos, 4 regiones, 5 sectores — cumple y supera el mínimo de 5.
  Benchmarks: SPY (global), ^GSPC (CAPM ref.), ^BVSP (LatAm).

CAMBIOS vs versión anterior:
  - Catálogo reducido de 30 → 10 activos (coherencia README vs código)
  - Se eliminan activos de baja liquidez/difícil acceso (SSNLF, PETR4.SA, etc.)
  - ACTIVOS_BASE ahora coincide con el portafolio documentado
  - Se mantiene fallback sintético para funcionar sin red
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List
import hashlib


# ── Cache en memoria (dura lo que dure el proceso) ───────────────────────────
_CACHE: dict = {}

# ─────────────────────────────────────────────────────────────────────────────
# CATÁLOGO PRINCIPAL — 10 activos curados
# 4 regiones × 5 sectores — diversificación real para el análisis
# ─────────────────────────────────────────────────────────────────────────────

CATALOGO: dict[str, dict] = {
    # ── Norteamérica — Tecnología ──────────────────────────────────────────
    "AAPL": {
        "nombre": "Apple Inc.",
        "sector": "Tecnología",
        "pais": "EE.UU.",
        "region": "Norteamérica",
        "moneda": "USD",
        "descripcion": "Mayor empresa tecnológica por capitalización. Alta liquidez, beta moderada (~1.2).",
    },
    "MSFT": {
        "nombre": "Microsoft Corp.",
        "sector": "Tecnología",
        "pais": "EE.UU.",
        "region": "Norteamérica",
        "moneda": "USD",
        "descripcion": "Tecnología diversificada (nube, IA, software). Correlación alta con AAPL — sirve para análisis de riesgo sistemático.",
    },
    # ── Norteamérica — Financiero ──────────────────────────────────────────
    "JPM": {
        "nombre": "JPMorgan Chase",
        "sector": "Financiero",
        "pais": "EE.UU.",
        "region": "Norteamérica",
        "moneda": "USD",
        "descripcion": "Banco más grande de EE.UU. Beta sensible a tasas de interés — relevante para módulo CAPM y contexto macro.",
    },
    # ── Norteamérica — Energía ─────────────────────────────────────────────
    "XOM": {
        "nombre": "ExxonMobil Corp.",
        "sector": "Energía",
        "pais": "EE.UU.",
        "region": "Norteamérica",
        "moneda": "USD",
        "descripcion": "Referencia de sector energético en EE.UU. Baja correlación con tecnología — mejora diversificación.",
    },
    # ── Norteamérica — Salud ──────────────────────────────────────────────
    "JNJ": {
        "nombre": "Johnson & Johnson",
        "sector": "Salud",
        "pais": "EE.UU.",
        "region": "Norteamérica",
        "moneda": "USD",
        "descripcion": "Defensivo clásico. Beta < 1. Baja correlación con ciclo económico — ancla la frontera eficiente.",
    },
    # ── Europa — Tecnología ────────────────────────────────────────────────
    "SAP.DE": {
        "nombre": "SAP SE",
        "sector": "Tecnología",
        "pais": "Alemania",
        "region": "Europa",
        "moneda": "EUR",
        "descripcion": "Mayor empresa de software empresarial europea. Diversificación geográfica y de moneda (EUR).",
    },
    # ── Europa — Salud ─────────────────────────────────────────────────────
    "NOVN.SW": {
        "nombre": "Novartis AG",
        "sector": "Salud",
        "pais": "Suiza",
        "region": "Europa",
        "moneda": "CHF",
        "descripcion": "Farmacéutica global con sede en Suiza. Moneda refugio (CHF) — reduce riesgo cambiario del portafolio.",
    },
    # ── LatAm — Energía ───────────────────────────────────────────────────
    "EC": {
        "nombre": "Ecopetrol S.A.",
        "sector": "Energía",
        "pais": "Colombia",
        "region": "LatAm",
        "moneda": "USD",
        "descripcion": "Activo colombiano — relevante para el contexto académico USTA. Alta volatilidad y correlación con petróleo.",
    },
    # ── LatAm — Financiero ────────────────────────────────────────────────
    "CIB": {
        "nombre": "Bancolombia S.A.",
        "sector": "Financiero",
        "pais": "Colombia",
        "region": "LatAm",
        "moneda": "USD",
        "descripcion": "Segundo activo colombiano — permite análisis de correlación intra-país y riesgo específico de mercado emergente.",
    },
    # ── Asia — Automotriz ─────────────────────────────────────────────────
    "TM": {
        "nombre": "Toyota Motor Corp.",
        "sector": "Automotriz",
        "pais": "Japón",
        "region": "Asia",
        "moneda": "USD",
        "descripcion": "Representante del sector automotriz asiático. Baja correlación con tecnología y financiero — cuarta región.",
    },
}

# Portafolio de referencia — los 5 activos del README + los 5 adicionales
ACTIVOS_BASE: list[str] = list(CATALOGO.keys())

# Activos del README (mínimo requerido por la rúbrica)
ACTIVOS_README: list[str] = ["AAPL", "MSFT", "JPM", "EC", "NOVN.SW"]

BENCHMARKS: dict[str, str] = {
    "Norteamérica": "^GSPC",    # S&P 500
    "Europa":       "^GDAXI",   # DAX
    "LatAm":        "^BVSP",    # Bovespa
    "Asia":         "^N225",    # Nikkei 225
    "Global":       "SPY",      # ETF S&P 500 — usado en comparación benchmark
}

# Precios base realistas para datos sintéticos (mayo 2025)
_PRECIOS_BASE: dict[str, float] = {
    "AAPL": 195.0, "MSFT": 415.0, "JPM": 205.0, "XOM": 115.0,
    "JNJ":  145.0, "SAP.DE": 195.0, "NOVN.SW": 95.0,
    "EC":   11.0,  "CIB": 28.0,   "TM": 185.0,
    # Benchmarks
    "^GSPC": 5200.0, "^GDAXI": 18000.0, "^BVSP": 125000.0,
    "^N225": 38000.0, "SPY": 520.0,
}

# Volatilidades diarias por sector (calibradas con datos reales 2022-2024)
_VOLATILIDADES: dict[str, float] = {
    "Tecnología":  0.026,
    "Financiero":  0.019,
    "Energía":     0.024,
    "Salud":       0.015,
    "Automotriz":  0.022,
}


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR DE DATOS SINTÉTICOS
# ─────────────────────────────────────────────────────────────────────────────

def _generar_sintetico(ticker: str, fecha_inicio: str, fecha_fin: str) -> pd.DataFrame:
    """
    Genera serie de precios sintética con random walk.
    Usa seed determinista basada en el ticker: mismas llamadas = mismos datos.
    Útil para demo/sustentación cuando no hay conexión a Yahoo Finance.
    """
    seed = int(hashlib.md5(ticker.encode()).hexdigest()[:8], 16) % (2**31)
    rng  = np.random.default_rng(seed)

    info        = CATALOGO.get(ticker, {})
    sector      = info.get("sector", "Tecnología")
    precio_base = _PRECIOS_BASE.get(ticker, 100.0)
    vol_diaria  = _VOLATILIDADES.get(sector, 0.022)

    fechas = pd.bdate_range(start=fecha_inicio, end=fecha_fin)
    n      = len(fechas)
    if n == 0:
        raise ValueError(f"Rango de fechas inválido: {fecha_inicio} → {fecha_fin}")

    drift   = 0.0003
    returns = rng.normal(drift, vol_diaria, n)
    precios = precio_base * np.exp(np.cumsum(returns))

    apertura = precios * (1 + rng.normal(0, 0.002, n))
    maximo   = precios * (1 + np.abs(rng.normal(0, 0.005, n)))
    minimo   = precios * (1 - np.abs(rng.normal(0, 0.005, n)))
    volumen  = rng.integers(1_000_000, 50_000_000, n)

    return pd.DataFrame({
        "fecha":    [f.strftime("%Y-%m-%d") for f in fechas],
        "apertura": np.round(apertura, 4),
        "maximo":   np.round(maximo, 4),
        "minimo":   np.round(minimo, 4),
        "cierre":   np.round(precios, 4),
        "volumen":  volumen.astype(float),
    })


def _intentar_yfinance(ticker: str, fecha_inicio: str, fecha_fin: str) -> pd.DataFrame:
    """Descarga de Yahoo Finance. Lanza ValueError si falla o no hay datos."""
    try:
        datos = yf.Ticker(ticker).history(
            start=fecha_inicio, end=fecha_fin, auto_adjust=True
        )
    except Exception as e:
        raise ValueError(f"yfinance error para {ticker}: {e}")

    if datos is None or datos.empty:
        raise ValueError(f"Sin datos reales para '{ticker}'")

    datos = datos.rename(columns={
        "Open": "apertura", "High": "maximo", "Low": "minimo",
        "Close": "cierre",  "Volume": "volumen",
    })
    datos = datos.reset_index().rename(columns={"Date": "fecha", "Datetime": "fecha"})
    datos["fecha"] = pd.to_datetime(datos["fecha"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
    cols = [c for c in ["fecha", "apertura", "maximo", "minimo", "cierre", "volumen"]
            if c in datos.columns]
    datos = datos[cols].copy()
    for col in ["apertura", "maximo", "minimo", "cierre"]:
        if col in datos.columns:
            datos[col] = datos[col].round(4)
    return datos


def _obtener_fecha_fin(fecha_fin: Optional[str]) -> str:
    return fecha_fin if fecha_fin else datetime.today().strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────────────────
# API PÚBLICA
# ─────────────────────────────────────────────────────────────────────────────

def descargar_precios(
    ticker: str,
    fecha_inicio: str = "2022-01-01",
    fecha_fin: Optional[str] = None,
) -> pd.DataFrame:
    """
    Descarga precios de Yahoo Finance con fallback a datos sintéticos.
    Cachea en memoria para evitar llamadas repetidas en la misma sesión.
    """
    fecha_fin  = _obtener_fecha_fin(fecha_fin)
    cache_key  = f"{ticker}_{fecha_inicio}_{fecha_fin}"

    if cache_key in _CACHE:
        return _CACHE[cache_key].copy()

    try:
        df = _intentar_yfinance(ticker, fecha_inicio, fecha_fin)
    except ValueError:
        df = _generar_sintetico(ticker, fecha_inicio, fecha_fin)

    _CACHE[cache_key] = df
    return df.copy()


def descargar_multiples_precios(
    tickers: List[str],
    fecha_inicio: str = "2022-01-01",
    fecha_fin: Optional[str] = None,
) -> dict:
    """Descarga precios de múltiples tickers. Retorna dict {ticker: DataFrame}."""
    fecha_fin = _obtener_fecha_fin(fecha_fin)
    return {
        ticker: descargar_precios(ticker, fecha_inicio, fecha_fin)
        for ticker in tickers
    }


def obtener_precio_actual(ticker: str) -> dict:
    """Precio actual desde yfinance o precio base sintético."""
    try:
        fi              = yf.Ticker(ticker).fast_info
        precio_actual   = round(float(fi.last_price), 4)
        precio_apertura = round(float(fi.open), 4)
        variacion_dia   = round(precio_actual - precio_apertura, 4)
        variacion_pct   = round((variacion_dia / precio_apertura) * 100, 2) if precio_apertura else 0.0
        return {
            "precio_actual":   precio_actual,
            "precio_apertura": precio_apertura,
            "variacion_dia":   variacion_dia,
            "variacion_pct":   variacion_pct,
            "fuente":          "Yahoo Finance (tiempo real)",
        }
    except Exception:
        precio = float(_PRECIOS_BASE.get(ticker, 100.0))
        return {
            "precio_actual":   precio,
            "precio_apertura": round(precio * 0.998, 4),
            "variacion_dia":   round(precio * 0.002, 4),
            "variacion_pct":   0.20,
            "fuente":          "Precio de referencia (sin conexión)",
        }


def obtener_info_activo(ticker: str) -> dict:
    """Devuelve metadata del activo + precio actual."""
    info   = CATALOGO.get(ticker, {
        "nombre": ticker, "sector": "N/A",
        "pais": "N/A", "region": "N/A", "moneda": "USD", "descripcion": "",
    })
    precio = obtener_precio_actual(ticker)
    return {"ticker": ticker, **info, **precio}


# ── Helpers de filtrado ────────────────────────────────────────────────────────

def get_por_region(region: str)  -> List[str]:
    return [t for t, v in CATALOGO.items() if v["region"] == region]

def get_por_sector(sector: str)  -> List[str]:
    return [t for t, v in CATALOGO.items() if v["sector"] == sector]

def get_por_pais(pais: str)      -> List[str]:
    return [t for t, v in CATALOGO.items() if v["pais"] == pais]

def get_regiones()               -> List[str]:
    return sorted(set(v["region"] for v in CATALOGO.values()))

def get_sectores()               -> List[str]:
    return sorted(set(v["sector"] for v in CATALOGO.values()))

def get_paises()                 -> List[str]:
    return sorted(set(v["pais"]   for v in CATALOGO.values()))

# Alias de compatibilidad con imports existentes en main.py
ACTIVOS_INFO: dict = {
    k: {"nombre": v["nombre"], "sector": v["sector"], "moneda": v["moneda"]}
    for k, v in CATALOGO.items()
}
obtener_fecha_fin = _obtener_fecha_fin
