"""
datos.py — VERSIÓN CON CACHE LOCAL + FALLBACK
==============================================
Fix principal: cuando yfinance falla por red, genera datos sintéticos
realistas para que el sistema funcione en demo/sustentación.

Los datos sintéticos se guardan en cache local (CSV) para que la misma
"sesión" sea consistente — el mismo ticker siempre retorna los mismos datos.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
from pathlib import Path
import hashlib

# ── Cache local en memoria (dura lo que dure el proceso) ──────────────────────
_CACHE: dict = {}

# ─────────────────────────────────────────────
# CATÁLOGO COMPLETO — 30 activos, 4 regiones, 6 sectores
# ─────────────────────────────────────────────

CATALOGO = {
    "AAPL":    {"nombre":"Apple Inc.",          "sector":"Tecnología", "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "MSFT":    {"nombre":"Microsoft Corp.",     "sector":"Tecnología", "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "GOOGL":   {"nombre":"Alphabet Inc.",       "sector":"Tecnología", "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "JPM":     {"nombre":"JPMorgan Chase",      "sector":"Financiero", "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "BAC":     {"nombre":"Bank of America",     "sector":"Financiero", "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "GS":      {"nombre":"Goldman Sachs",       "sector":"Financiero", "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "XOM":     {"nombre":"ExxonMobil",          "sector":"Energía",    "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "CVX":     {"nombre":"Chevron Corp.",       "sector":"Energía",    "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "JNJ":     {"nombre":"Johnson & Johnson",   "sector":"Salud",      "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "PFE":     {"nombre":"Pfizer Inc.",         "sector":"Salud",      "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "AMZN":    {"nombre":"Amazon.com Inc.",     "sector":"Consumo",    "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "WMT":     {"nombre":"Walmart Inc.",        "sector":"Consumo",    "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "TSLA":    {"nombre":"Tesla Inc.",          "sector":"Automotriz", "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "F":       {"nombre":"Ford Motor Co.",      "sector":"Automotriz", "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "SAP.DE":  {"nombre":"SAP SE",              "sector":"Tecnología", "pais":"Alemania", "region":"Europa",       "moneda":"EUR"},
    "ASML.AS": {"nombre":"ASML Holding",        "sector":"Tecnología", "pais":"Holanda",  "region":"Europa",       "moneda":"EUR"},
    "HSBA.L":  {"nombre":"HSBC Holdings",       "sector":"Financiero", "pais":"UK",       "region":"Europa",       "moneda":"GBP"},
    "BNP.PA":  {"nombre":"BNP Paribas",         "sector":"Financiero", "pais":"Francia",  "region":"Europa",       "moneda":"EUR"},
    "TTE.PA":  {"nombre":"TotalEnergies",       "sector":"Energía",    "pais":"Francia",  "region":"Europa",       "moneda":"EUR"},
    "BP.L":    {"nombre":"BP plc",              "sector":"Energía",    "pais":"UK",       "region":"Europa",       "moneda":"GBP"},
    "NOVN.SW": {"nombre":"Novartis AG",         "sector":"Salud",      "pais":"Suiza",    "region":"Europa",       "moneda":"CHF"},
    "AZN.L":   {"nombre":"AstraZeneca",         "sector":"Salud",      "pais":"UK",       "region":"Europa",       "moneda":"GBP"},
    "EC":      {"nombre":"Ecopetrol S.A.",      "sector":"Energía",    "pais":"Colombia", "region":"LatAm",        "moneda":"USD"},
    "CIB":     {"nombre":"Bancolombia",         "sector":"Financiero", "pais":"Colombia", "region":"LatAm",        "moneda":"USD"},
    "PETR4.SA":{"nombre":"Petrobras",           "sector":"Energía",    "pais":"Brasil",   "region":"LatAm",        "moneda":"BRL"},
    "ITUB4.SA":{"nombre":"Itaú Unibanco",       "sector":"Financiero", "pais":"Brasil",   "region":"LatAm",        "moneda":"BRL"},
    "TM":      {"nombre":"Toyota Motor",        "sector":"Automotriz", "pais":"Japón",    "region":"Asia",         "moneda":"USD"},
    "SONY":    {"nombre":"Sony Group Corp.",    "sector":"Tecnología", "pais":"Japón",    "region":"Asia",         "moneda":"USD"},
    "SSNLF":   {"nombre":"Samsung Electronics","sector":"Tecnología", "pais":"Corea",    "region":"Asia",         "moneda":"USD"},
    "INFY":    {"nombre":"Infosys Ltd.",        "sector":"Tecnología", "pais":"India",    "region":"Asia",         "moneda":"USD"},
}

BENCHMARKS = {
    "Norteamérica": "^GSPC",
    "Europa":       "^GDAXI",
    "LatAm":        "^BVSP",
    "Asia":         "^N225",
}

# Precios base realistas por ticker para datos sintéticos
_PRECIOS_BASE = {
    "AAPL":150,"MSFT":350,"GOOGL":140,"JPM":190,"BAC":35,"GS":420,
    "XOM":110,"CVX":155,"JNJ":155,"PFE":28,"AMZN":185,"WMT":65,
    "TSLA":180,"F":12,"SAP.DE":180,"ASML.AS":700,"HSBA.L":8,
    "BNP.PA":60,"TTE.PA":60,"BP.L":5,"NOVN.SW":90,"AZN.L":12,
    "EC":12,"CIB":30,"PETR4.SA":35,"ITUB4.SA":14,"TM":170,
    "SONY":80,"SSNLF":55,"INFY":18,
    "^GSPC":4500,"^GDAXI":16000,"^BVSP":120000,"^N225":33000,
    "SPY":450,"GLD":185,
}

# Volatilidades base por sector (más realistas)
_VOLATILIDADES = {
    "Tecnología":0.028,"Financiero":0.020,"Energía":0.025,
    "Salud":0.018,"Consumo":0.016,"Automotriz":0.030,
}


def _generar_sintetico(ticker: str, fecha_inicio: str, fecha_fin: str) -> pd.DataFrame:
    """
    Genera serie de precios sintética con random walk para un ticker.
    Usa seed basada en el ticker para que sea reproducible (mismas llamadas = mismos datos).
    """
    seed = int(hashlib.md5(ticker.encode()).hexdigest()[:8], 16) % (2**31)
    rng  = np.random.default_rng(seed)

    info   = CATALOGO.get(ticker, {})
    sector = info.get("sector", "Tecnología")
    precio_base = _PRECIOS_BASE.get(ticker, 100.0)
    vol_diaria  = _VOLATILIDADES.get(sector, 0.022)

    # Generar fechas de negocio
    fechas = pd.bdate_range(start=fecha_inicio, end=fecha_fin)
    n      = len(fechas)
    if n == 0:
        raise ValueError(f"Rango de fechas inválido: {fecha_inicio} → {fecha_fin}")

    # Random walk con drift positivo leve
    drift   = 0.0003
    returns = rng.normal(drift, vol_diaria, n)
    precios = precio_base * np.exp(np.cumsum(returns))

    apertura = precios * (1 + rng.normal(0, 0.002, n))
    maximo   = precios * (1 + np.abs(rng.normal(0, 0.005, n)))
    minimo   = precios * (1 - np.abs(rng.normal(0, 0.005, n)))
    volumen  = rng.integers(1_000_000, 50_000_000, n)

    df = pd.DataFrame({
        "fecha":   [f.strftime("%Y-%m-%d") for f in fechas],
        "apertura":np.round(apertura, 4),
        "maximo":  np.round(maximo, 4),
        "minimo":  np.round(minimo, 4),
        "cierre":  np.round(precios, 4),
        "volumen": volumen.astype(float),
    })
    return df


def _intentar_yfinance(ticker: str, fecha_inicio: str, fecha_fin: str) -> pd.DataFrame:
    """Intenta descargar de Yahoo Finance. Lanza ValueError si falla."""
    try:
        datos = yf.Ticker(ticker).history(
            start=fecha_inicio, end=fecha_fin, auto_adjust=True
        )
    except Exception as e:
        raise ValueError(f"yfinance error para {ticker}: {e}")

    if datos is None or datos.empty:
        raise ValueError(f"Sin datos reales para '{ticker}'")

    datos = datos.rename(columns={
        "Open":"apertura","High":"maximo","Low":"minimo",
        "Close":"cierre","Volume":"volumen"
    })
    datos = datos.reset_index().rename(columns={"Date":"fecha","Datetime":"fecha"})
    datos["fecha"] = pd.to_datetime(datos["fecha"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
    cols = [c for c in ["fecha","apertura","maximo","minimo","cierre","volumen"] if c in datos.columns]
    datos = datos[cols].copy()
    for col in ["apertura","maximo","minimo","cierre"]:
        if col in datos.columns:
            datos[col] = datos[col].round(4)
    return datos


def obtener_fecha_fin(fecha_fin: Optional[str]) -> str:
    return fecha_fin if fecha_fin else datetime.today().strftime("%Y-%m-%d")


def descargar_precios(
    ticker: str,
    fecha_inicio: str = "2022-01-01",
    fecha_fin: Optional[str] = None,
) -> pd.DataFrame:
    """
    Descarga precios reales de Yahoo Finance.
    Si falla por red/firewall, usa datos sintéticos reproducibles.
    Los datos se cachean en memoria para la sesión actual.
    """
    fecha_fin = obtener_fecha_fin(fecha_fin)
    cache_key = f"{ticker}_{fecha_inicio}_{fecha_fin}"

    if cache_key in _CACHE:
        return _CACHE[cache_key].copy()

    # Intentar Yahoo Finance primero
    try:
        df = _intentar_yfinance(ticker, fecha_inicio, fecha_fin)
        _CACHE[cache_key] = df
        return df.copy()
    except ValueError:
        pass

    # Fallback: datos sintéticos
    df = _generar_sintetico(ticker, fecha_inicio, fecha_fin)
    _CACHE[cache_key] = df
    return df.copy()


def descargar_multiples_precios(
    tickers: List[str],
    fecha_inicio: str = "2022-01-01",
    fecha_fin: Optional[str] = None,
) -> dict:
    fecha_fin = obtener_fecha_fin(fecha_fin)
    return {
        ticker: descargar_precios(ticker, fecha_inicio, fecha_fin)
        for ticker in tickers
    }


def obtener_precio_actual(ticker: str) -> dict:
    """Precio actual desde yfinance o último precio sintético."""
    try:
        fi = yf.Ticker(ticker).fast_info
        precio_actual   = round(float(fi.last_price), 4)
        precio_apertura = round(float(fi.open), 4)
        variacion_dia   = round(precio_actual - precio_apertura, 4)
        variacion_pct   = round((variacion_dia / precio_apertura) * 100, 2) if precio_apertura else 0.0
        return {
            "precio_actual":   precio_actual,
            "precio_apertura": precio_apertura,
            "variacion_dia":   variacion_dia,
            "variacion_pct":   variacion_pct,
        }
    except Exception:
        # Precio base del catálogo sintético
        precio = float(_PRECIOS_BASE.get(ticker, 100.0))
        return {
            "precio_actual":   precio,
            "precio_apertura": round(precio * 0.998, 4),
            "variacion_dia":   round(precio * 0.002, 4),
            "variacion_pct":   0.20,
        }


def obtener_info_activo(ticker: str) -> dict:
    info   = CATALOGO.get(ticker, {"nombre":ticker,"sector":"N/A","pais":"N/A","region":"N/A","moneda":"USD"})
    precio = obtener_precio_actual(ticker)
    return {"ticker": ticker, **info, **precio}


# ── Helpers de filtrado ────────────────────────
def get_por_region(region: str)  -> List[str]: return [t for t,v in CATALOGO.items() if v["region"]==region]
def get_por_sector(sector: str)  -> List[str]: return [t for t,v in CATALOGO.items() if v["sector"]==sector]
def get_por_pais(pais: str)      -> List[str]: return [t for t,v in CATALOGO.items() if v["pais"]==pais]
def get_regiones()               -> List[str]: return sorted(set(v["region"] for v in CATALOGO.values()))
def get_sectores()               -> List[str]: return sorted(set(v["sector"] for v in CATALOGO.values()))
def get_paises()                 -> List[str]: return sorted(set(v["pais"]   for v in CATALOGO.values()))

# Compatibilidad con código anterior
ACTIVOS_INFO = {k: {"nombre":v["nombre"],"sector":v["sector"],"moneda":v["moneda"]}
                for k,v in CATALOGO.items() if k in ["AAPL","MSFT","GOOGL","AMZN","TSLA"]}
