import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List

# ─────────────────────────────────────────────
# CATÁLOGO COMPLETO — 30 activos, 4 regiones, 6 sectores
# ─────────────────────────────────────────────

CATALOGO = {
    # ── EE.UU. ────────────────────────────────
    "AAPL":  {"nombre":"Apple Inc.",         "sector":"Tecnología", "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "MSFT":  {"nombre":"Microsoft Corp.",    "sector":"Tecnología", "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "GOOGL": {"nombre":"Alphabet Inc.",      "sector":"Tecnología", "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "JPM":   {"nombre":"JPMorgan Chase",     "sector":"Financiero", "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "BAC":   {"nombre":"Bank of America",    "sector":"Financiero", "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "GS":    {"nombre":"Goldman Sachs",      "sector":"Financiero", "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "XOM":   {"nombre":"ExxonMobil",         "sector":"Energía",    "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "CVX":   {"nombre":"Chevron Corp.",      "sector":"Energía",    "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "JNJ":   {"nombre":"Johnson & Johnson",  "sector":"Salud",      "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "PFE":   {"nombre":"Pfizer Inc.",        "sector":"Salud",      "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "AMZN":  {"nombre":"Amazon.com Inc.",    "sector":"Consumo",    "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "WMT":   {"nombre":"Walmart Inc.",       "sector":"Consumo",    "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "TSLA":  {"nombre":"Tesla Inc.",         "sector":"Automotriz", "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    "F":     {"nombre":"Ford Motor Co.",     "sector":"Automotriz", "pais":"EE.UU.",   "region":"Norteamérica", "moneda":"USD"},
    # ── Europa ────────────────────────────────
    "SAP.DE":  {"nombre":"SAP SE",           "sector":"Tecnología", "pais":"Alemania", "region":"Europa",       "moneda":"EUR"},
    "ASML.AS": {"nombre":"ASML Holding",     "sector":"Tecnología", "pais":"Holanda",  "region":"Europa",       "moneda":"EUR"},
    "HSBA.L":  {"nombre":"HSBC Holdings",    "sector":"Financiero", "pais":"UK",       "region":"Europa",       "moneda":"GBP"},
    "BNP.PA":  {"nombre":"BNP Paribas",      "sector":"Financiero", "pais":"Francia",  "region":"Europa",       "moneda":"EUR"},
    "TTE.PA":  {"nombre":"TotalEnergies",    "sector":"Energía",    "pais":"Francia",  "region":"Europa",       "moneda":"EUR"},
    "BP.L":    {"nombre":"BP plc",           "sector":"Energía",    "pais":"UK",       "region":"Europa",       "moneda":"GBP"},
    "NOVN.SW": {"nombre":"Novartis AG",      "sector":"Salud",      "pais":"Suiza",    "region":"Europa",       "moneda":"CHF"},
    "AZN.L":   {"nombre":"AstraZeneca",      "sector":"Salud",      "pais":"UK",       "region":"Europa",       "moneda":"GBP"},
    # ── América Latina ────────────────────────
    "EC":       {"nombre":"Ecopetrol S.A.",  "sector":"Energía",    "pais":"Colombia", "region":"LatAm",        "moneda":"USD"},
    "CIB":      {"nombre":"Bancolombia",     "sector":"Financiero", "pais":"Colombia", "region":"LatAm",        "moneda":"USD"},
    "PETR4.SA": {"nombre":"Petrobras",       "sector":"Energía",    "pais":"Brasil",   "region":"LatAm",        "moneda":"BRL"},
    "ITUB4.SA": {"nombre":"Itaú Unibanco",   "sector":"Financiero", "pais":"Brasil",   "region":"LatAm",        "moneda":"BRL"},
    # ── Asia ──────────────────────────────────
    "TM":    {"nombre":"Toyota Motor",       "sector":"Automotriz", "pais":"Japón",    "region":"Asia",         "moneda":"USD"},
    "SONY":  {"nombre":"Sony Group Corp.",   "sector":"Tecnología", "pais":"Japón",    "region":"Asia",         "moneda":"USD"},
    "SSNLF": {"nombre":"Samsung Electronics","sector":"Tecnología", "pais":"Corea",    "region":"Asia",         "moneda":"USD"},
    "INFY":  {"nombre":"Infosys Ltd.",       "sector":"Tecnología", "pais":"India",    "region":"Asia",         "moneda":"USD"},
    "SFTBY": {"nombre":"SoftBank Group",     "sector":"Tecnología", "pais":"Japón",    "region":"Asia",         "moneda":"USD"},
}

BENCHMARKS = {
    "Norteamérica": "^GSPC",
    "Europa":       "^GDAXI",
    "LatAm":        "^BVSP",
    "Asia":         "^N225",
}

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


# ─────────────────────────────────────────────
# DESCARGA DE PRECIOS
# ─────────────────────────────────────────────

def obtener_fecha_fin(fecha_fin: Optional[str]) -> str:
    return fecha_fin if fecha_fin else datetime.today().strftime("%Y-%m-%d")


def descargar_precios(
    ticker: str,
    fecha_inicio: str = "2022-01-01",
    fecha_fin: Optional[str] = None,
) -> pd.DataFrame:
    fecha_fin = obtener_fecha_fin(fecha_fin)
    try:
        datos = yf.Ticker(ticker).history(start=fecha_inicio, end=fecha_fin, auto_adjust=True)
    except Exception as e:
        raise ValueError(f"Error Yahoo Finance para {ticker}: {e}")

    if datos is None or datos.empty:
        raise ValueError(f"Sin datos para '{ticker}' entre {fecha_inicio} y {fecha_fin}.")

    datos = datos.rename(columns={"Open":"apertura","High":"maximo","Low":"minimo","Close":"cierre","Volume":"volumen"})
    datos = datos.reset_index().rename(columns={"Date":"fecha","Datetime":"fecha"})
    datos["fecha"] = pd.to_datetime(datos["fecha"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")

    cols = [c for c in ["fecha","apertura","maximo","minimo","cierre","volumen"] if c in datos.columns]
    datos = datos[cols].copy()
    for col in ["apertura","maximo","minimo","cierre"]:
        if col in datos.columns:
            datos[col] = datos[col].round(4)
    return datos


def descargar_multiples_precios(
    tickers: List[str],
    fecha_inicio: str = "2022-01-01",
    fecha_fin: Optional[str] = None,
) -> dict:
    fecha_fin = obtener_fecha_fin(fecha_fin)
    resultado = {}
    for ticker in tickers:
        try:
            resultado[ticker] = descargar_precios(ticker, fecha_inicio, fecha_fin)
        except ValueError as e:
            resultado[ticker] = None
            print(f"Advertencia: {e}")
    return resultado


def obtener_precio_actual(ticker: str) -> dict:
    try:
        fi = yf.Ticker(ticker).fast_info
        precio_actual   = round(float(fi.last_price), 4)
        precio_apertura = round(float(fi.open), 4)
        variacion_dia   = round(precio_actual - precio_apertura, 4)
        variacion_pct   = round((variacion_dia / precio_apertura) * 100, 2) if precio_apertura else 0.0
        return {"precio_actual":precio_actual,"precio_apertura":precio_apertura,
                "variacion_dia":variacion_dia,"variacion_pct":variacion_pct}
    except Exception:
        return {"precio_actual":None,"precio_apertura":None,"variacion_dia":None,"variacion_pct":None}


def obtener_info_activo(ticker: str) -> dict:
    info  = CATALOGO.get(ticker, {"nombre":ticker,"sector":"N/A","pais":"N/A","region":"N/A","moneda":"USD"})
    precio = obtener_precio_actual(ticker)
    return {"ticker": ticker, **info, **precio}
