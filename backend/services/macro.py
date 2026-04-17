import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
from services.datos import descargar_precios, ACTIVOS_INFO
from services.indicadores import (
    calcular_rsi,
    calcular_ema,
    calcular_macd,
    calcular_bollinger,
    calcular_estocastico,
)


def _limpiar(v):
    """Convierte tipos numpy a tipos nativos de Python."""
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


# ─────────────────────────────────────────────
# SEÑALES Y ALERTAS DE TRADING
# ─────────────────────────────────────────────

def generar_alertas_portafolio(
    tickers: List[str] = None,
    fecha_inicio: str = "2023-01-01",
) -> dict:
    """
    Genera señales automáticas de compra/venta para todos los activos
    basadas en múltiples indicadores técnicos.

    Lógica de señales:
    - RSI < 30         → COMPRA fuerte (sobrevendido)
    - RSI > 70         → VENTA fuerte  (sobrecomprado)
    - MACD > Señal     → COMPRA (tendencia alcista)
    - MACD < Señal     → VENTA  (tendencia bajista)
    - Precio < BB inf  → COMPRA (precio bajo banda)
    - Precio > BB sup  → VENTA  (precio sobre banda)
    - EMA20 > EMA50    → COMPRA (golden cross corto plazo)
    - EMA20 < EMA50    → VENTA  (death cross corto plazo)
    - Estocástico < 20 → COMPRA (sobrevendido)
    - Estocástico > 80 → VENTA  (sobrecomprado)
    """
    if tickers is None:
        tickers = list(ACTIVOS_INFO.keys())

    todas_alertas = []
    resumen_por_ticker = {}

    for ticker in tickers:
        try:
            df = descargar_precios(ticker, fecha_inicio)
            precios = df["cierre"]
            alertas_ticker = []

            # ── Calcular indicadores del último día ───────────────────────────
            rsi    = calcular_rsi(precios)
            ema20  = calcular_ema(precios, 20)
            ema50  = calcular_ema(precios, 50)
            macd_df = calcular_macd(precios)
            boll_df = calcular_bollinger(precios)
            esto_df = calcular_estocastico(df)

            # Último valor de cada indicador
            rsi_actual    = _limpiar(rsi.iloc[-1])
            ema20_actual  = _limpiar(ema20.iloc[-1])
            ema50_actual  = _limpiar(ema50.iloc[-1])
            macd_actual   = _limpiar(macd_df["macd"].iloc[-1])
            señal_actual  = _limpiar(macd_df["macd_señal"].iloc[-1])
            boll_sup      = _limpiar(boll_df["boll_superior"].iloc[-1])
            boll_inf      = _limpiar(boll_df["boll_inferior"].iloc[-1])
            precio_actual = _limpiar(precios.iloc[-1])
            esto_k        = _limpiar(esto_df["esto_k"].iloc[-1])

            # ── Señal RSI ─────────────────────────────────────────────────────
            if rsi_actual is not None:
                if rsi_actual < 30:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "COMPRA", "RSI", "FUERTE",
                        f"RSI={rsi_actual:.1f} — sobrevendido (< 30)",
                        rsi_actual
                    ))
                elif rsi_actual < 40:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "COMPRA", "RSI", "DÉBIL",
                        f"RSI={rsi_actual:.1f} — acercándose a zona de compra",
                        rsi_actual
                    ))
                elif rsi_actual > 70:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "VENTA", "RSI", "FUERTE",
                        f"RSI={rsi_actual:.1f} — sobrecomprado (> 70)",
                        rsi_actual
                    ))
                elif rsi_actual > 60:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "VENTA", "RSI", "DÉBIL",
                        f"RSI={rsi_actual:.1f} — acercándose a zona de venta",
                        rsi_actual
                    ))

            # ── Señal MACD ────────────────────────────────────────────────────
            if macd_actual is not None and señal_actual is not None:
                diferencia = round(macd_actual - señal_actual, 4)
                if macd_actual > señal_actual:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "COMPRA", "MACD", "MODERADA",
                        f"MACD ({macd_actual:.4f}) sobre línea señal ({señal_actual:.4f})",
                        diferencia
                    ))
                else:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "VENTA", "MACD", "MODERADA",
                        f"MACD ({macd_actual:.4f}) bajo línea señal ({señal_actual:.4f})",
                        diferencia
                    ))

            # ── Señal Bollinger ───────────────────────────────────────────────
            if precio_actual and boll_sup and boll_inf:
                if precio_actual > boll_sup:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "VENTA", "Bollinger", "FUERTE",
                        f"Precio ({precio_actual:.2f}) sobre banda superior ({boll_sup:.2f})",
                        precio_actual
                    ))
                elif precio_actual < boll_inf:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "COMPRA", "Bollinger", "FUERTE",
                        f"Precio ({precio_actual:.2f}) bajo banda inferior ({boll_inf:.2f})",
                        precio_actual
                    ))

            # ── Señal EMA (Golden/Death Cross) ────────────────────────────────
            if ema20_actual and ema50_actual:
                if ema20_actual > ema50_actual:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "COMPRA", "EMA_Cross", "MODERADA",
                        f"EMA20 ({ema20_actual:.2f}) sobre EMA50 ({ema50_actual:.2f}) — tendencia alcista",
                        ema20_actual - ema50_actual
                    ))
                else:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "VENTA", "EMA_Cross", "MODERADA",
                        f"EMA20 ({ema20_actual:.2f}) bajo EMA50 ({ema50_actual:.2f}) — tendencia bajista",
                        ema20_actual - ema50_actual
                    ))

            # ── Señal Estocástico ─────────────────────────────────────────────
            if esto_k is not None:
                if esto_k < 20:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "COMPRA", "Estocástico", "FUERTE",
                        f"%K={esto_k:.1f} — zona de sobrevendido (< 20)",
                        esto_k
                    ))
                elif esto_k > 80:
                    alertas_ticker.append(_crear_alerta(
                        ticker, "VENTA", "Estocástico", "FUERTE",
                        f"%K={esto_k:.1f} — zona de sobrecomprado (> 80)",
                        esto_k
                    ))

            # ── Resumen por ticker ────────────────────────────────────────────
            compras = sum(1 for a in alertas_ticker if a["tipo"] == "COMPRA")
            ventas  = sum(1 for a in alertas_ticker if a["tipo"] == "VENTA")
            señal_neta = "NEUTRAL"
            if compras > ventas + 1:
                señal_neta = "COMPRA"
            elif ventas > compras + 1:
                señal_neta = "VENTA"

            resumen_por_ticker[ticker] = {
                "precio_actual":  precio_actual,
                "rsi_actual":     rsi_actual,
                "señal_neta":     señal_neta,
                "alertas_compra": compras,
                "alertas_venta":  ventas,
                "fecha_analisis": datetime.today().strftime("%Y-%m-%d"),
            }

            todas_alertas.extend(alertas_ticker)

        except Exception as e:
            resumen_por_ticker[ticker] = {"error": str(e)}

    # Ordenar alertas: FUERTE primero
    prioridad = {"FUERTE": 0, "MODERADA": 1, "DÉBIL": 2}
    todas_alertas.sort(key=lambda x: prioridad.get(x.get("fuerza", "DÉBIL"), 2))

    return {
        "fecha_analisis":  datetime.today().strftime("%Y-%m-%d %H:%M"),
        "tickers_analizados": tickers,
        "total_alertas":   len(todas_alertas),
        "alertas_compra":  sum(1 for a in todas_alertas if a["tipo"] == "COMPRA"),
        "alertas_venta":   sum(1 for a in todas_alertas if a["tipo"] == "VENTA"),
        "resumen":         resumen_por_ticker,
        "alertas":         todas_alertas,
    }


def _crear_alerta(
    ticker: str,
    tipo: str,
    indicador: str,
    fuerza: str,
    descripcion: str,
    valor,
) -> dict:
    """Crea un diccionario de alerta estandarizado."""
    return {
        "ticker":      ticker,
        "tipo":        tipo,           # COMPRA o VENTA
        "indicador":   indicador,      # RSI, MACD, Bollinger, etc.
        "fuerza":      fuerza,         # FUERTE, MODERADA, DÉBIL
        "descripcion": descripcion,
        "valor":       _limpiar(valor),
        "fecha":       datetime.today().strftime("%Y-%m-%d"),
    }


# ─────────────────────────────────────────────
# DATOS MACROECONÓMICOS — FRED API
# ─────────────────────────────────────────────

# Series FRED más importantes para análisis de riesgo
SERIES_FRED = {
    "DGS3MO":  "Tasa libre de riesgo (T-Bills 3 meses)",
    "DGS10":   "Tasa del Tesoro a 10 años",
    "CPIAUCSL": "Índice de Precios al Consumidor (CPI)",
    "UNRATE":  "Tasa de desempleo",
    "GDP":     "Producto Interno Bruto",
    "FEDFUNDS": "Tasa de fondos federales",
    "VIXCLS":  "Índice VIX (volatilidad del mercado)",
    "SP500":   "S&P 500",
}


def obtener_datos_fred(
    api_key: Optional[str] = None,
    series: List[str] = None,
) -> dict:
    """
    Obtiene indicadores macroeconómicos desde la API de FRED.

    Si no hay API key, retorna datos de ejemplo con valores recientes.

    Para obtener tu API key gratis:
    1. Ve a https://fred.stlouisfed.org/docs/api/api_key.html
    2. Crea una cuenta gratuita
    3. Solicita una API key
    4. Agrégala al archivo .env como FRED_API_KEY=tu_clave
    """
    if series is None:
        series = ["DGS3MO", "DGS10", "CPIAUCSL", "UNRATE", "FEDFUNDS", "VIXCLS"]

    # Si no hay API key, retornar datos de ejemplo
    if not api_key:
        return _datos_fred_ejemplo()

    resultados = {}
    errores    = []

    for serie_id in series:
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id":       serie_id,
                "api_key":         api_key,
                "file_type":       "json",
                "sort_order":      "desc",
                "limit":           5,           # últimos 5 registros
                "observation_start": (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d"),
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            observaciones = data.get("observations", [])
            if observaciones:
                # Tomar el valor más reciente que no sea "."
                valor_reciente = None
                fecha_reciente = None
                for obs in observaciones:
                    if obs["value"] != ".":
                        valor_reciente = float(obs["value"])
                        fecha_reciente = obs["date"]
                        break

                resultados[serie_id] = {
                    "nombre":      SERIES_FRED.get(serie_id, serie_id),
                    "valor":       valor_reciente,
                    "fecha":       fecha_reciente,
                    "unidad":      "%" if serie_id not in ["GDP", "SP500", "VIXCLS"] else "índice",
                }
        except Exception as e:
            errores.append(f"{serie_id}: {str(e)}")

    if errores:
        resultados["_errores"] = errores

    return {
        "fuente":      "FRED - Federal Reserve Bank of St. Louis",
        "fecha_consulta": datetime.today().strftime("%Y-%m-%d %H:%M"),
        "datos":       resultados,
    }


def _datos_fred_ejemplo() -> dict:
    """
    Retorna datos macroeconómicos de ejemplo cuando no hay API key.
    Valores aproximados de inicios de 2025.
    """
    return {
        "fuente":         "FRED - Federal Reserve Bank of St. Louis",
        "fecha_consulta": datetime.today().strftime("%Y-%m-%d %H:%M"),
        "nota":           "Datos de ejemplo. Configura FRED_API_KEY en .env para datos reales.",
        "como_obtener_key": "https://fred.stlouisfed.org/docs/api/api_key.html",
        "datos": {
            "DGS3MO": {
                "nombre": "Tasa libre de riesgo (T-Bills 3 meses)",
                "valor":  5.25,
                "fecha":  "2025-01-01",
                "unidad": "%",
                "uso_en_capm": "Se usa como Rf en la fórmula CAPM: E(R) = Rf + Beta*(Rm-Rf)",
            },
            "DGS10": {
                "nombre": "Tasa del Tesoro a 10 años",
                "valor":  4.58,
                "fecha":  "2025-01-01",
                "unidad": "%",
            },
            "CPIAUCSL": {
                "nombre": "Inflación anual (CPI)",
                "valor":  3.2,
                "fecha":  "2025-01-01",
                "unidad": "%",
            },
            "UNRATE": {
                "nombre": "Tasa de desempleo",
                "valor":  3.9,
                "fecha":  "2025-01-01",
                "unidad": "%",
            },
            "FEDFUNDS": {
                "nombre": "Tasa de fondos federales (Fed)",
                "valor":  5.33,
                "fecha":  "2025-01-01",
                "unidad": "%",
            },
            "VIXCLS": {
                "nombre": "VIX — Índice de volatilidad del mercado",
                "valor":  15.4,
                "fecha":  "2025-01-01",
                "unidad": "índice",
                "interpretacion": (
                    "VIX < 20: baja volatilidad (mercado tranquilo). "
                    "VIX 20-30: volatilidad moderada. "
                    "VIX > 30: alta volatilidad (pánico en el mercado)."
                ),
            },
        },
        "contexto_macro": {
            "descripcion": (
                "Las tasas altas de la Fed (5.33%) aumentan el costo de capital "
                "y presionan las valoraciones de las acciones de crecimiento. "
                "Con inflación en 3.2%, el rendimiento real de los T-Bills es ~2%, "
                "lo que hace que los bonos compitan con las acciones."
            ),
            "impacto_portafolio": [
                "Tasa libre de riesgo alta → prima de riesgo de acciones se reduce",
                "Inflación moderada → la Fed podría bajar tasas gradualmente",
                "VIX bajo → mercado en modo 'risk-on', favorable para acciones",
                "Desempleo bajo → economía sólida, buenos fundamentos corporativos",
            ],
        },
    }
