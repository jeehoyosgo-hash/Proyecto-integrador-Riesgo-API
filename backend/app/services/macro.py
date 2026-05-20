"""
macro.py — Señales de trading y datos macroeconómicos FRED
===========================================================
CORRECCIÓN principal: generar_alertas_portafolio ahora acepta
rsi_sobrecompra y rsi_sobreventa como parámetros y los usa en
la lógica de señales (requisito explícito del Módulo 7 — umbrales configurables).
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List

from app.services.datos import descargar_precios, ACTIVOS_INFO
from app.services.indicadores import (
    calcular_rsi, calcular_ema, calcular_macd,
    calcular_bollinger, calcular_estocastico,
)


def _limpiar(v):
    if v is None:
        return None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 4)
    if isinstance(v, float):
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    return v


# ─────────────────────────────────────────────────────────────────────────────
# SEÑALES Y ALERTAS — MÓDULO 7
# ─────────────────────────────────────────────────────────────────────────────

def generar_alertas_portafolio(
    tickers: List[str] = None,
    fecha_inicio: str = "2023-01-01",
    rsi_sobrecompra: int = 70,    # ← NUEVO: umbral configurable
    rsi_sobreventa: int = 30,     # ← NUEVO: umbral configurable
) -> dict:
    """
    Genera señales automáticas de compra/venta para cada activo del portafolio.

    Señales implementadas (Módulo 7 completo):
    1. RSI:          < rsi_sobreventa → COMPRA FUERTE; > rsi_sobrecompra → VENTA FUERTE
    2. MACD:         línea MACD vs. señal (cruce alcista/bajista)
    3. Bollinger:    precio fuera de bandas
    4. EMA Cross:    EMA20 vs EMA50 (golden/death cross)
    5. Estocástico:  %K en zonas extremas

    Los umbrales RSI son configurables via parámetros (no hardcodeados).
    """
    if tickers is None:
        tickers = list(ACTIVOS_INFO.keys())

    todas_alertas: list = []
    resumen_por_ticker: dict = {}

    for ticker in tickers:
        try:
            df            = descargar_precios(ticker, fecha_inicio)
            precios       = df["cierre"]
            alertas_ticker: list = []

            # ── Calcular todos los indicadores ────────────────────────────────
            rsi_series  = calcular_rsi(precios)
            ema20       = calcular_ema(precios, 20)
            ema50       = calcular_ema(precios, 50)
            macd_df     = calcular_macd(precios)
            boll_df     = calcular_bollinger(precios)
            esto_df     = calcular_estocastico(df)

            # Último valor de cada indicador
            rsi_actual    = _limpiar(rsi_series.iloc[-1])
            ema20_actual  = _limpiar(ema20.iloc[-1])
            ema50_actual  = _limpiar(ema50.iloc[-1])
            macd_actual   = _limpiar(macd_df["macd"].iloc[-1])
            señal_actual  = _limpiar(macd_df["macd_señal"].iloc[-1])
            boll_sup      = _limpiar(boll_df["boll_superior"].iloc[-1])
            boll_inf      = _limpiar(boll_df["boll_inferior"].iloc[-1])
            precio_actual = _limpiar(precios.iloc[-1])
            esto_k        = _limpiar(esto_df["esto_k"].iloc[-1])

            # ── 1. Señal RSI (umbrales configurables) ─────────────────────────
            zona_compra_fuerte = rsi_sobreventa
            zona_compra_debil  = rsi_sobreventa + 10
            zona_venta_fuerte  = rsi_sobrecompra
            zona_venta_debil   = rsi_sobrecompra - 10

            if rsi_actual is not None:
                if rsi_actual < zona_compra_fuerte:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "COMPRA", "RSI", "FUERTE",
                        f"RSI={rsi_actual:.1f} — sobrevendido (< {zona_compra_fuerte})",
                        rsi_actual,
                    ))
                elif rsi_actual < zona_compra_debil:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "COMPRA", "RSI", "DÉBIL",
                        f"RSI={rsi_actual:.1f} — acercándose a zona de compra (< {zona_compra_debil})",
                        rsi_actual,
                    ))
                elif rsi_actual > zona_venta_fuerte:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "VENTA", "RSI", "FUERTE",
                        f"RSI={rsi_actual:.1f} — sobrecomprado (> {zona_venta_fuerte})",
                        rsi_actual,
                    ))
                elif rsi_actual > zona_venta_debil:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "VENTA", "RSI", "DÉBIL",
                        f"RSI={rsi_actual:.1f} — acercándose a zona de venta (> {zona_venta_debil})",
                        rsi_actual,
                    ))

            # ── 2. Señal MACD (cruce de líneas) ──────────────────────────────
            if macd_actual is not None and señal_actual is not None:
                diferencia = round(macd_actual - señal_actual, 4)
                if macd_actual > señal_actual:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "COMPRA", "MACD", "MODERADA",
                        f"MACD ({macd_actual:.4f}) sobre señal ({señal_actual:.4f}) — tendencia alcista",
                        diferencia,
                    ))
                else:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "VENTA", "MACD", "MODERADA",
                        f"MACD ({macd_actual:.4f}) bajo señal ({señal_actual:.4f}) — tendencia bajista",
                        diferencia,
                    ))

            # ── 3. Señal Bandas de Bollinger ──────────────────────────────────
            if precio_actual and boll_sup and boll_inf:
                if precio_actual > boll_sup:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "VENTA", "Bollinger", "FUERTE",
                        f"Precio ({precio_actual:.2f}) sobre banda superior ({boll_sup:.2f})",
                        precio_actual,
                    ))
                elif precio_actual < boll_inf:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "COMPRA", "Bollinger", "FUERTE",
                        f"Precio ({precio_actual:.2f}) bajo banda inferior ({boll_inf:.2f})",
                        precio_actual,
                    ))

            # ── 4. Señal EMA Cross (golden / death cross) ─────────────────────
            if ema20_actual and ema50_actual:
                diferencia_ema = round(ema20_actual - ema50_actual, 4)
                if ema20_actual > ema50_actual:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "COMPRA", "EMA_Cross", "MODERADA",
                        f"Golden cross: EMA20 ({ema20_actual:.2f}) > EMA50 ({ema50_actual:.2f})",
                        diferencia_ema,
                    ))
                else:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "VENTA", "EMA_Cross", "MODERADA",
                        f"Death cross: EMA20 ({ema20_actual:.2f}) < EMA50 ({ema50_actual:.2f})",
                        diferencia_ema,
                    ))

            # ── 5. Señal Oscilador Estocástico ────────────────────────────────
            if esto_k is not None:
                if esto_k < 20:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "COMPRA", "Estocástico", "FUERTE",
                        f"%K={esto_k:.1f} — zona de sobrevendido (< 20)",
                        esto_k,
                    ))
                elif esto_k > 80:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "VENTA", "Estocástico", "FUERTE",
                        f"%K={esto_k:.1f} — zona de sobrecomprado (> 80)",
                        esto_k,
                    ))

            # ── Resumen semáforo por ticker ───────────────────────────────────
            compras    = sum(1 for a in alertas_ticker if a["tipo"] == "COMPRA")
            ventas     = sum(1 for a in alertas_ticker if a["tipo"] == "VENTA")
            señal_neta = "NEUTRAL"
            if compras > ventas + 1:
                señal_neta = "COMPRA"
            elif ventas > compras + 1:
                señal_neta = "VENTA"

            resumen_por_ticker[ticker] = {
                "precio_actual":  precio_actual,
                "rsi_actual":     rsi_actual,
                "señal_neta":     señal_neta,    # semáforo
                "alertas_compra": compras,
                "alertas_venta":  ventas,
                "fecha_analisis": datetime.today().strftime("%Y-%m-%d"),
            }
            todas_alertas.extend(alertas_ticker)

        except Exception as e:
            resumen_por_ticker[ticker] = {"error": str(e)}

    # Ordenar: FUERTE primero
    prioridad = {"FUERTE": 0, "MODERADA": 1, "DÉBIL": 2}
    todas_alertas.sort(key=lambda x: prioridad.get(x.get("fuerza", "DÉBIL"), 2))

    return {
        "fecha_analisis":      datetime.today().strftime("%Y-%m-%d %H:%M"),
        "tickers_analizados":  tickers,
        "total_alertas":       len(todas_alertas),
        "alertas_compra":      sum(1 for a in todas_alertas if a["tipo"] == "COMPRA"),
        "alertas_venta":       sum(1 for a in todas_alertas if a["tipo"] == "VENTA"),
        "umbrales_rsi":        {"sobrecompra": rsi_sobrecompra, "sobreventa": rsi_sobreventa},
        "resumen":             resumen_por_ticker,
        "alertas":             todas_alertas,
    }


def _crear_alerta(
    ticker: str,
    tipo: str,
    indicador: str,
    fuerza: str,
    descripcion: str,
    valor,
) -> dict:
    return {
        "ticker":      ticker,
        "tipo":        tipo,
        "indicador":   indicador,
        "fuerza":      fuerza,
        "descripcion": descripcion,
        "valor":       _limpiar(valor),
        "fecha":       datetime.today().strftime("%Y-%m-%d"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DATOS MACROECONÓMICOS — FRED API (Módulo 8)
# ─────────────────────────────────────────────────────────────────────────────

SERIES_FRED: dict[str, str] = {
    "DGS3MO":   "Tasa libre de riesgo (T-Bills 3 meses)",
    "DGS10":    "Tasa del Tesoro a 10 años",
    "CPIAUCSL": "Índice de Precios al Consumidor (CPI)",
    "UNRATE":   "Tasa de desempleo",
    "FEDFUNDS": "Tasa de fondos federales (Fed)",
    "VIXCLS":   "VIX — Índice de volatilidad del mercado",
}


def obtener_datos_fred(
    api_key: Optional[str] = None,
    series: List[str] = None,
) -> dict:
    """
    Obtiene indicadores macroeconómicos desde FRED.
    CPI se calcula como variación anual (no índice acumulado).
    Retorna datos de referencia si no hay API key.
    """
    if series is None:
        series = list(SERIES_FRED.keys())

    if not api_key:
        return _datos_fred_ejemplo()

    resultados: dict = {}
    errores: list    = []

    for serie_id in series:
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            if serie_id == "CPIAUCSL":
                params = {
                    "series_id": serie_id, "api_key": api_key,
                    "file_type": "json", "sort_order": "desc", "limit": 16,
                }
            else:
                params = {
                    "series_id":         serie_id, "api_key": api_key,
                    "file_type":         "json", "sort_order": "desc", "limit": 5,
                    "observation_start": (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d"),
                }
            resp          = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            observaciones = [o for o in resp.json().get("observations", []) if o["value"] != "."]
            if not observaciones:
                continue

            if serie_id == "CPIAUCSL":
                valor_reciente = (
                    round((float(observaciones[0]["value"]) / float(observaciones[12]["value"]) - 1) * 100, 2)
                    if len(observaciones) >= 13 else None
                )
            else:
                valor_reciente = float(observaciones[0]["value"])

            interpretacion = _interpretar_indicador(serie_id, valor_reciente)

            resultados[serie_id] = {
                "nombre":         SERIES_FRED.get(serie_id, serie_id),
                "valor":          valor_reciente,
                "fecha":          observaciones[0]["date"],
                "unidad":         "pts" if serie_id == "VIXCLS" else "%",
                "interpretacion": interpretacion,
                "aplica_a":       "Global" if serie_id == "VIXCLS" else "EE.UU.",
            }
        except Exception as e:
            errores.append(f"{serie_id}: {str(e)}")

    if errores:
        resultados["_errores"] = errores

    contexto = _generar_contexto_macro(resultados)

    return {
        "fuente":         "FRED — Federal Reserve Bank of St. Louis",
        "fecha_consulta": datetime.today().strftime("%Y-%m-%d %H:%M"),
        "nota_alcance":   (
            "DGS3MO, DGS10, CPI, UNRATE y FEDFUNDS corresponden a EE.UU. "
            "El VIX es global. El T-Bills 3M se usa como Rf en el CAPM."
        ),
        "datos":          resultados,
        "contexto_macro": contexto,
    }


def _interpretar_indicador(serie_id: str, valor) -> str:
    if valor is None:
        return "Sin datos disponibles."
    if serie_id == "DGS3MO":
        if valor > 5:   return f"Tasa alta ({valor:.2f}%). Encarece el costo de capital. Usado como Rf en CAPM."
        elif valor > 3: return f"Tasa moderada ({valor:.2f}%). Referencia libre de riesgo para el CAPM."
        else:           return f"Tasa baja ({valor:.2f}%). Favorable para acciones — costo de oportunidad bajo."
    if serie_id == "DGS10":
        if valor > 4.5: return f"Rendimiento alto ({valor:.2f}%). Los bonos compiten con las acciones."
        elif valor > 3: return f"Rendimiento moderado ({valor:.2f}%). Entorno neutral."
        else:           return f"Rendimiento bajo ({valor:.2f}%). Favorable para acciones de crecimiento."
    if serie_id == "CPIAUCSL":
        if valor > 5:   return f"Inflación alta ({valor:.2f}%). La Fed mantiene tasas elevadas."
        elif valor > 2.5: return f"Inflación moderada ({valor:.2f}%). Por encima del objetivo del 2%."
        else:           return f"Inflación controlada ({valor:.2f}%). Cerca del objetivo del 2%."
    if serie_id == "UNRATE":
        if valor > 6:   return f"Desempleo alto ({valor:.2f}%). Señal de debilidad económica."
        elif valor > 4.5: return f"Desempleo moderado ({valor:.2f}%). Mercado laboral en recuperación."
        else:           return f"Desempleo bajo ({valor:.2f}%). Economía sólida."
    if serie_id == "FEDFUNDS":
        if valor > 5:   return f"Tasa Fed alta ({valor:.2f}%). Política restrictiva."
        elif valor > 3: return f"Tasa Fed moderada ({valor:.2f}%). Política neutral."
        else:           return f"Tasa Fed baja ({valor:.2f}%). Política expansiva."
    if serie_id == "VIXCLS":
        if valor > 30:  return f"VIX alto ({valor:.1f}). Pánico en el mercado — VaR puede subestimar el riesgo."
        elif valor > 20: return f"VIX moderado ({valor:.1f}). Incertidumbre — precaución recomendada."
        else:           return f"VIX bajo ({valor:.1f}). Mercado tranquilo — modo risk-on."
    return "—"


def _generar_contexto_macro(datos: dict) -> dict:
    fed    = datos.get("FEDFUNDS", {}).get("valor")
    cpi    = datos.get("CPIAUCSL", {}).get("valor")
    vix    = datos.get("VIXCLS",   {}).get("valor")
    unrate = datos.get("UNRATE",   {}).get("valor")
    rf     = datos.get("DGS3MO",   {}).get("valor")

    partes = []
    if fed and cpi:
        partes.append(
            f"La Fed mantiene su tasa en {fed:.2f}% con inflación en {cpi:.2f}% anual. "
            f"Tasa real (Fed - CPI): {fed - cpi:.2f}%."
        )
    if vix:
        estado = "tranquilo" if vix < 20 else "con incertidumbre" if vix < 30 else "en pánico"
        partes.append(f"El mercado está {estado} — VIX en {vix:.1f} puntos.")
    if unrate:
        partes.append(f"Desempleo en EE.UU.: {unrate:.1f}%.")

    impacto = []
    if rf:
        impacto.append(f"Rf en CAPM = {rf:.2f}% (T-Bills 3M)")
    if vix and vix > 25:
        impacto.append("VIX elevado → el VaR histórico puede subestimar el riesgo real")
    if fed and fed > 4:
        impacto.append("Tasas altas → acciones de alto P/E (tecnología) más vulnerables")
    if cpi and cpi > 3:
        impacto.append(f"Inflación ({cpi:.1f}%) > 2% → Fed podría mantener tasas restrictivas")

    return {
        "descripcion":        " ".join(partes) if partes else "Datos macroeconómicos actualizados.",
        "impacto_portafolio": impacto if impacto else ["Entorno macroeconómico estable."],
    }


def _datos_fred_ejemplo() -> dict:
    datos = {
        "DGS3MO":   {"nombre": "Tasa libre de riesgo (T-Bills 3M)", "valor": 4.35, "fecha": "2025-04-01", "unidad": "%", "aplica_a": "EE.UU.", "interpretacion": "Tasa moderada. Referencia Rf para CAPM."},
        "DGS10":    {"nombre": "Tasa del Tesoro a 10 años",          "valor": 4.26, "fecha": "2025-04-01", "unidad": "%", "aplica_a": "EE.UU.", "interpretacion": "Rendimiento alto. Los bonos compiten con acciones."},
        "CPIAUCSL": {"nombre": "Inflación anual (CPI)",              "valor": 2.80, "fecha": "2025-03-01", "unidad": "%", "aplica_a": "EE.UU.", "interpretacion": "Inflación moderada. Por encima del objetivo 2%."},
        "UNRATE":   {"nombre": "Tasa de desempleo",                  "valor": 4.20, "fecha": "2025-03-01", "unidad": "%", "aplica_a": "EE.UU.", "interpretacion": "Desempleo moderado. Mercado laboral en recuperación."},
        "FEDFUNDS": {"nombre": "Tasa de fondos federales",           "valor": 4.33, "fecha": "2025-03-01", "unidad": "%", "aplica_a": "EE.UU.", "interpretacion": "Política monetaria neutral."},
        "VIXCLS":   {"nombre": "VIX — Índice de volatilidad",        "valor": 18.87,"fecha": "2025-04-20", "unidad": "pts","aplica_a": "Global","interpretacion": "VIX bajo. Mercado tranquilo — modo risk-on."},
    }
    contexto = _generar_contexto_macro(datos)
    return {
        "fuente":         "FRED — Federal Reserve Bank of St. Louis",
        "fecha_consulta": datetime.today().strftime("%Y-%m-%d %H:%M"),
        "nota":           "Datos de referencia. Configura FRED_API_KEY en .env para datos en tiempo real.",
        "nota_alcance":   "T-Bills 3M se usa como Rf en el CAPM.",
        "datos":          datos,
        "contexto_macro": contexto,
    }
