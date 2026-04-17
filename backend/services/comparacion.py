import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, List
from services.datos import (
    descargar_precios, descargar_multiples_precios,
    CATALOGO, get_por_region, get_por_sector
)
from services.indicadores import calcular_rsi, calcular_ema, calcular_macd


def _limpiar(v):
    if v is None: return None
    if isinstance(v, (np.bool_,)): return bool(v)
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    if isinstance(v, float):
        return None if (np.isnan(v) or np.isinf(v)) else v
    return v

def _limpiar_dict(d):
    if isinstance(d, dict): return {k: _limpiar_dict(v) for k, v in d.items()}
    if isinstance(d, list): return [_limpiar_dict(i) for i in d]
    return _limpiar(d)


# ─────────────────────────────────────────────
# COMPARACIÓN DE ACTIVOS
# ─────────────────────────────────────────────

def comparar_activos(
    tickers: List[str],
    fecha_inicio: str = "2022-01-01",
    fecha_fin: Optional[str] = None,
) -> dict:
    """
    Compara múltiples activos lado a lado con métricas clave:
    rendimiento, volatilidad, Sharpe, RSI, tendencia, etc.
    Permite comparar activos de distintos sectores, países y regiones.
    """
    datos = descargar_multiples_precios(tickers, fecha_inicio, fecha_fin)
    comparacion = {}
    errores = []

    for ticker in tickers:
        if datos[ticker] is None:
            errores.append(ticker)
            continue

        df     = datos[ticker]
        precios = df["cierre"]

        # Rendimientos logarítmicos
        rend = np.log(precios / precios.shift(1)).dropna()

        media_diaria  = float(rend.mean())
        std_diaria    = float(rend.std())
        media_anual   = media_diaria * 252
        std_anual     = std_diaria * np.sqrt(252)
        sharpe        = (media_anual - 0.0525) / std_anual if std_anual > 0 else 0
        max_dd        = _max_drawdown(precios)

        # Retorno total del período
        retorno_total = (precios.iloc[-1] - precios.iloc[0]) / precios.iloc[0]

        # Momentum 3 meses (63 días de trading)
        if len(precios) >= 63:
            momentum_3m = (precios.iloc[-1] - precios.iloc[-63]) / precios.iloc[-63]
        else:
            momentum_3m = retorno_total

        # RSI actual
        rsi_series = calcular_rsi(precios)
        rsi_actual = _limpiar(rsi_series.iloc[-1])

        # Tendencia EMA
        ema20 = calcular_ema(precios, 20)
        ema50 = calcular_ema(precios, 50)
        tendencia = "alcista" if float(ema20.iloc[-1] or 0) > float(ema50.iloc[-1] or 0) else "bajista"

        info = CATALOGO.get(ticker, {})

        comparacion[ticker] = {
            "ticker":              ticker,
            "nombre":              info.get("nombre", ticker),
            "sector":              info.get("sector", "N/A"),
            "pais":                info.get("pais", "N/A"),
            "region":              info.get("region", "N/A"),
            "moneda":              info.get("moneda", "USD"),
            "precio_actual":       _limpiar(float(precios.iloc[-1])),
            "precio_inicio":       _limpiar(float(precios.iloc[0])),
            "retorno_total":       _limpiar(retorno_total),
            "retorno_total_pct":   f"{retorno_total*100:.2f}%",
            "retorno_anual":       _limpiar(media_anual),
            "retorno_anual_pct":   f"{media_anual*100:.2f}%",
            "volatilidad_anual":   _limpiar(std_anual),
            "volatilidad_anual_pct": f"{std_anual*100:.2f}%",
            "sharpe_ratio":        _limpiar(sharpe),
            "max_drawdown":        _limpiar(max_dd),
            "max_drawdown_pct":    f"{max_dd*100:.2f}%",
            "momentum_3m":         _limpiar(momentum_3m),
            "momentum_3m_pct":     f"{momentum_3m*100:.2f}%",
            "rsi_actual":          rsi_actual,
            "tendencia_ema":       tendencia,
            "observaciones":       len(df),
        }

    # Ranking por Sharpe Ratio
    activos_validos = [(t, v["sharpe_ratio"] or 0) for t, v in comparacion.items()]
    activos_validos.sort(key=lambda x: x[1], reverse=True)
    ranking = {t: i+1 for i, (t, _) in enumerate(activos_validos)}
    for ticker in comparacion:
        comparacion[ticker]["ranking_sharpe"] = ranking.get(ticker, 0)

    return _limpiar_dict({
        "tickers":         tickers,
        "fecha_inicio":    fecha_inicio,
        "fecha_fin":       fecha_fin or "hoy",
        "total_activos":   len(comparacion),
        "errores":         errores,
        "comparacion":     comparacion,
        "mejor_sharpe":    activos_validos[0][0] if activos_validos else None,
        "mejor_retorno":   max(comparacion.items(), key=lambda x: x[1]["retorno_total"] or -99)[0] if comparacion else None,
        "menor_volatilidad": min(comparacion.items(), key=lambda x: x[1]["volatilidad_anual"] or 99)[0] if comparacion else None,
    })


def _max_drawdown(precios: pd.Series) -> float:
    """Calcula el máximo drawdown (caída máxima desde un pico)."""
    pico = precios.expanding().max()
    caida = (precios - pico) / pico
    return float(caida.min())


# ─────────────────────────────────────────────
# MOTOR DE RECOMENDACIONES
# ─────────────────────────────────────────────

def recomendar_portafolio(
    tickers: List[str] = None,
    perfil_riesgo: str = "moderado",
    fecha_inicio: str = "2022-01-01",
    fecha_fin: Optional[str] = None,
    tasa_libre_riesgo: float = 0.0525,
) -> dict:
    """
    Motor de recomendaciones basado en scoring multifactor.

    Scoring ponderado:
      40% — Sharpe Ratio (mejor balance riesgo/retorno)
      25% — Señales técnicas (RSI, MACD, EMA)
      20% — Momentum 3 meses
      15% — Diversificación (penaliza concentración en un sector/región)

    Perfiles de riesgo:
      conservador  → prefiere baja volatilidad y activos defensivos
      moderado     → balance entre retorno y riesgo
      agresivo     → maximiza retorno esperado

    Retorna portafolio recomendado con pesos y justificación.
    """
    if tickers is None:
        # Por defecto, un activo representativo de cada sector/región
        tickers = ["AAPL","MSFT","JPM","XOM","JNJ","AMZN","EC","CIB","TM","INFY","NOVN.SW","HSBA.L"]

    # 1. Descargar datos y calcular métricas
    datos = descargar_multiples_precios(tickers, fecha_inicio, fecha_fin)
    scores = {}

    for ticker in tickers:
        if datos[ticker] is None:
            continue

        df     = datos[ticker]
        precios = df["cierre"]

        if len(precios) < 60:
            continue

        rend = np.log(precios / precios.shift(1)).dropna()
        media_anual = float(rend.mean() * 252)
        std_anual   = float(rend.std() * np.sqrt(252))
        sharpe      = (media_anual - tasa_libre_riesgo) / std_anual if std_anual > 0 else 0

        # Momentum 3 meses
        momentum = float((precios.iloc[-1] - precios.iloc[-63]) / precios.iloc[-63]) if len(precios) >= 63 else 0

        # Score técnico: RSI, MACD, EMA
        rsi    = calcular_rsi(precios)
        ema20  = calcular_ema(precios, 20)
        ema50  = calcular_ema(precios, 50)
        macd_df = calcular_macd(precios)

        rsi_val  = float(rsi.iloc[-1] or 50)
        ema20_val = float(ema20.iloc[-1] or 0)
        ema50_val = float(ema50.iloc[-1] or 0)
        macd_val  = float(macd_df["macd"].iloc[-1] or 0)
        señal_val = float(macd_df["macd_señal"].iloc[-1] or 0)

        # Puntaje técnico: suma de señales positivas
        score_tecnico = 0
        if rsi_val < 50:   score_tecnico += 1   # RSI < 50 → no sobrecomprado
        if rsi_val > 30:   score_tecnico += 0.5  # RSI > 30 → no sobrevendido
        if ema20_val > ema50_val: score_tecnico += 1  # tendencia alcista
        if macd_val > señal_val:  score_tecnico += 1  # MACD positivo
        score_tecnico = score_tecnico / 3.5  # normalizar a 0-1

        # Max drawdown (penaliza caídas grandes)
        pico  = precios.expanding().max()
        max_dd = abs(float(((precios - pico) / pico).min()))

        scores[ticker] = {
            "sharpe":       sharpe,
            "momentum":     momentum,
            "score_tecnico": score_tecnico,
            "volatilidad":  std_anual,
            "max_drawdown": max_dd,
            "retorno":      media_anual,
        }

    if not scores:
        raise ValueError("No se pudieron calcular métricas para ningún activo.")

    # 2. Normalizar scores a escala 0-1
    def normalizar(vals, invertir=False):
        mn, mx = min(vals), max(vals)
        if mx == mn: return [0.5] * len(vals)
        norm = [(v - mn) / (mx - mn) for v in vals]
        return [1 - n for n in norm] if invertir else norm

    tickers_validos = list(scores.keys())
    sharpes    = [scores[t]["sharpe"]        for t in tickers_validos]
    momentums  = [scores[t]["momentum"]      for t in tickers_validos]
    tecnicos   = [scores[t]["score_tecnico"] for t in tickers_validos]
    vols       = [scores[t]["volatilidad"]   for t in tickers_validos]

    sharpes_n   = normalizar(sharpes)
    momentums_n = normalizar(momentums)
    tecnicos_n  = normalizar(tecnicos)
    vols_n      = normalizar(vols, invertir=True)  # menor volatilidad = mejor

    # 3. Pesos del scoring según perfil
    pesos_scoring = {
        "conservador": {"sharpe":0.30, "tecnico":0.20, "momentum":0.10, "vol":0.40},
        "moderado":    {"sharpe":0.40, "tecnico":0.25, "momentum":0.20, "vol":0.15},
        "agresivo":    {"sharpe":0.35, "tecnico":0.25, "momentum":0.35, "vol":0.05},
    }
    pw = pesos_scoring.get(perfil_riesgo, pesos_scoring["moderado"])

    # 4. Score final por activo
    scores_finales = {}
    for i, ticker in enumerate(tickers_validos):
        sf = (pw["sharpe"]   * sharpes_n[i] +
              pw["tecnico"]  * tecnicos_n[i] +
              pw["momentum"] * momentums_n[i] +
              pw["vol"]      * vols_n[i])
        scores_finales[ticker] = round(sf, 4)

    # 5. Seleccionar top activos para el portafolio (máx 8)
    top_n = 8
    ranking = sorted(scores_finales.items(), key=lambda x: x[1], reverse=True)[:top_n]
    tickers_seleccionados = [t for t, _ in ranking]

    # 6. Calcular pesos del portafolio proporcionales al score
    total_score = sum(s for _, s in ranking)
    pesos_raw   = {t: s / total_score for t, s in ranking}

    # Ajustar por perfil: conservador concentra menos, agresivo puede concentrar más
    max_peso = {"conservador": 0.20, "moderado": 0.25, "agresivo": 0.35}.get(perfil_riesgo, 0.25)
    pesos_ajustados = {}
    exceso = 0
    for t, w in pesos_raw.items():
        if w > max_peso:
            exceso += w - max_peso
            pesos_ajustados[t] = max_peso
        else:
            pesos_ajustados[t] = w

    # Redistribuir exceso proporcionalmente
    if exceso > 0:
        tickers_bajo_max = [t for t, w in pesos_ajustados.items() if w < max_peso]
        if tickers_bajo_max:
            extra_por_activo = exceso / len(tickers_bajo_max)
            for t in tickers_bajo_max:
                pesos_ajustados[t] = min(pesos_ajustados[t] + extra_por_activo, max_peso)

    # Normalizar para que sumen exactamente 1
    total = sum(pesos_ajustados.values())
    pesos_finales = {t: round(w / total, 4) for t, w in pesos_ajustados.items()}

    # 7. Diversificación por sector y región
    sectores_port  = {}
    regiones_port  = {}
    for ticker in tickers_seleccionados:
        info = CATALOGO.get(ticker, {})
        s = info.get("sector", "N/A")
        r = info.get("region", "N/A")
        sectores_port[s] = sectores_port.get(s, 0) + pesos_finales.get(ticker, 0)
        regiones_port[r] = regiones_port.get(r, 0) + pesos_finales.get(ticker, 0)

    # 8. Métricas del portafolio recomendado
    retorno_port = sum(scores[t]["retorno"] * pesos_finales[t] for t in tickers_seleccionados if t in scores)

    # 9. Justificación
    mejor_ticker = ranking[0][0]
    mejor_info   = CATALOGO.get(mejor_ticker, {})
    justificacion = _generar_justificacion(
        perfil_riesgo, ranking, scores, sectores_port, regiones_port, retorno_port
    )

    return _limpiar_dict({
        "perfil_riesgo":    perfil_riesgo,
        "fecha_analisis":   pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "portafolio_recomendado": {
            t: {
                "peso":          pesos_finales[t],
                "peso_pct":      f"{pesos_finales[t]*100:.1f}%",
                "score":         scores_finales[t],
                "nombre":        CATALOGO.get(t, {}).get("nombre", t),
                "sector":        CATALOGO.get(t, {}).get("sector", "N/A"),
                "pais":          CATALOGO.get(t, {}).get("pais", "N/A"),
                "region":        CATALOGO.get(t, {}).get("region", "N/A"),
                "sharpe":        round(scores[t]["sharpe"], 4),
                "retorno_anual": f"{scores[t]['retorno']*100:.2f}%",
                "volatilidad":   f"{scores[t]['volatilidad']*100:.2f}%",
                "momentum_3m":   f"{scores[t]['momentum']*100:.2f}%",
            }
            for t in tickers_seleccionados if t in scores
        },
        "metricas_portafolio": {
            "retorno_esperado_anual": round(retorno_port, 4),
            "retorno_esperado_pct":   f"{retorno_port*100:.2f}%",
            "n_activos":              len(tickers_seleccionados),
        },
        "diversificacion": {
            "por_sector":  {k: round(v, 4) for k, v in sectores_port.items()},
            "por_region":  {k: round(v, 4) for k, v in regiones_port.items()},
        },
        "ranking_completo": [
            {"ticker": t, "score": s, "nombre": CATALOGO.get(t,{}).get("nombre",t)}
            for t, s in ranking
        ],
        "justificacion": justificacion,
        "metodologia": {
            "descripcion": "Scoring multifactor ponderado",
            "factores": {
                "Sharpe Ratio":     f"{pw['sharpe']*100:.0f}%",
                "Señales técnicas": f"{pw['tecnico']*100:.0f}%",
                "Momentum 3M":      f"{pw['momentum']*100:.0f}%",
                "Baja volatilidad": f"{pw['vol']*100:.0f}%",
            }
        }
    })


def _generar_justificacion(
    perfil: str, ranking: list, scores: dict,
    sectores: dict, regiones: dict, retorno: float
) -> dict:
    """Genera una justificación estructurada de la recomendación."""

    mejor = ranking[0][0]
    peor  = ranking[-1][0] if len(ranking) > 1 else mejor
    sector_dominante  = max(sectores.items(),  key=lambda x: x[1])[0] if sectores else "N/A"
    region_dominante  = max(regiones.items(),  key=lambda x: x[1])[0] if regiones else "N/A"

    descripcion_perfil = {
        "conservador": "priorizando activos defensivos con baja volatilidad y flujo de caja estable",
        "moderado":    "balanceando retorno esperado con control de riesgo",
        "agresivo":    "maximizando el retorno esperado con mayor tolerancia al riesgo",
    }.get(perfil, "")

    alertas = []
    for ticker, _ in ranking:
        s = scores.get(ticker, {})
        if s.get("max_drawdown", 0) > 0.40:
            alertas.append(f"{ticker} tiene un drawdown histórico alto ({s['max_drawdown']*100:.1f}%) — riesgo de caída severa")
        if s.get("volatilidad", 0) > 0.45:
            alertas.append(f"{ticker} es muy volátil ({s['volatilidad']*100:.1f}% anual)")

    return {
        "resumen": (
            f"Para un perfil {perfil}, se recomienda un portafolio de {len(ranking)} activos "
            f"{descripcion_perfil}. El retorno esperado anual es {retorno*100:.2f}%."
        ),
        "activo_destacado": {
            "ticker":  mejor,
            "razon":   f"Mayor score combinado ({ranking[0][1]:.3f}) — mejor balance entre Sharpe Ratio, momentum y señales técnicas",
            "sharpe":  round(scores.get(mejor, {}).get("sharpe", 0), 3),
        },
        "diversificacion": (
            f"El portafolio concentra {sectores.get(sector_dominante,0)*100:.1f}% en {sector_dominante} "
            f"y {regiones.get(region_dominante,0)*100:.1f}% en {region_dominante}."
        ),
        "alertas_riesgo": alertas if alertas else ["No se detectaron alertas de riesgo significativas"],
        "recomendacion_accion": {
            "conservador": "Mantener posiciones defensivas. Revisar trimestralmente.",
            "moderado":    "Rebalancear semestralmente para mantener los pesos objetivo.",
            "agresivo":    "Monitorear señales técnicas semanalmente. Stop-loss recomendado en -15%.",
        }.get(perfil, "Revisar periódicamente."),
    }
