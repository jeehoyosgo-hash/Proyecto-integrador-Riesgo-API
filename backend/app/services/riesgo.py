import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, List
from app.services.datos import descargar_precios, descargar_multiples_precios


def _limpiar(v):
    """
    Convierte cualquier tipo numpy a tipo nativo de Python.
    FastAPI solo puede serializar a JSON tipos nativos: int, float, bool, str.
    numpy.bool_, numpy.float64, numpy.int64 causan errores si no se convierten.
    """
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
        return round(f, 6)
    if isinstance(v, float):
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    return v


def _limpiar_dict(d: dict) -> dict:
    """Aplica _limpiar() a todos los valores de un diccionario (recursivo)."""
    resultado = {}
    for k, v in d.items():
        if isinstance(v, dict):
            resultado[k] = _limpiar_dict(v)
        elif isinstance(v, list):
            resultado[k] = [_limpiar_dict(i) if isinstance(i, dict) else _limpiar(i) for i in v]
        else:
            resultado[k] = _limpiar(v)
    return resultado


# ─────────────────────────────────────────────
# RENDIMIENTOS
# ─────────────────────────────────────────────

def calcular_rendimientos(
    ticker: str,
    fecha_inicio: str = "2022-01-01",
    fecha_fin: Optional[str] = None,
) -> dict:
    """
    Calcula rendimientos simples y logarítmicos de un activo.

    Rendimiento simple:      (P_t - P_{t-1}) / P_{t-1}
    Rendimiento logarítmico: ln(P_t / P_{t-1})

    El logarítmico es preferido en finanzas porque:
    - Es aditivo en el tiempo
    - Tiene mejores propiedades estadísticas
    - Es simétrico (subidas y bajadas se tratan igual)
    """
    df          = descargar_precios(ticker, fecha_inicio, fecha_fin)
    precios     = df["cierre"]
    rend_simple = precios.pct_change().dropna()
    rend_log    = np.log(precios / precios.shift(1)).dropna()

    # Estadísticas descriptivas
    stats_simple = _estadisticas(rend_simple, "simple")
    stats_log    = _estadisticas(rend_log, "logarítmico")

    # Prueba Jarque-Bera
    jb_stat, jb_pvalue = stats.jarque_bera(rend_log)

    # Prueba Shapiro-Wilk (últimos 50 días)
    muestra           = rend_log.tail(50)
    sw_stat, sw_pvalue = stats.shapiro(muestra)

    # Serie de datos para la respuesta
    fechas = df["fecha"].iloc[1:].tolist()
    datos  = [
        {
            "fecha":       f,
            "rend_simple": round(float(rs), 6),
            "rend_log":    round(float(rl), 6),
        }
        for f, rs, rl in zip(fechas, rend_simple, rend_log)
    ]

    resultado = {
        "ticker":              ticker,
        "total_observaciones": len(datos),
        "estadisticas_simple": stats_simple,
        "estadisticas_log":    stats_log,
        "pruebas_normalidad": {
            "jarque_bera": {
                "estadistico":    float(jb_stat),
                "p_value":        float(jb_pvalue),
                "es_normal":      bool(jb_pvalue > 0.05),
                "interpretacion": (
                    "No se rechaza normalidad (p > 0.05)"
                    if jb_pvalue > 0.05
                    else "Se rechaza normalidad (p < 0.05) — distribución con colas pesadas"
                ),
            },
            "shapiro_wilk": {
                "estadistico": float(sw_stat),
                "p_value":     float(sw_pvalue),
                "es_normal":   bool(sw_pvalue > 0.05),
                "nota":        "Calculado sobre los últimos 50 días",
            },
        },
        "datos": datos,
    }

    # Limpiar todos los valores numpy antes de retornar
    return _limpiar_dict(resultado)


def _estadisticas(rendimientos: pd.Series, tipo: str) -> dict:
    """Estadísticas descriptivas de una serie de rendimientos."""
    return {
        "tipo":                       tipo,
        "media_diaria":               round(float(rendimientos.mean()), 6),
        "media_anual":                round(float(rendimientos.mean() * 252), 4),
        "volatilidad_diaria":         round(float(rendimientos.std()), 6),
        "volatilidad_anual":          round(float(rendimientos.std() * np.sqrt(252)), 4),
        "minimo":                     round(float(rendimientos.min()), 6),
        "maximo":                     round(float(rendimientos.max()), 6),
        "asimetria":                  round(float(rendimientos.skew()), 4),
        "curtosis":                   round(float(rendimientos.kurtosis()), 4),
        "interpretacion_curtosis":    (
            "Colas pesadas — más riesgo de eventos extremos que una normal"
            if rendimientos.kurtosis() > 0
            else "Colas ligeras — menos eventos extremos que una normal"
        ),
    }


# ─────────────────────────────────────────────
# VaR y CVaR
# ─────────────────────────────────────────────

def calcular_var_cvar(
    tickers: List[str],
    pesos: List[float],
    fecha_inicio: str = "2022-01-01",
    fecha_fin: Optional[str] = None,
    nivel_confianza: float = 0.95,
    valor_portafolio: float = 100_000,
) -> dict:
    """
    Calcula VaR y CVaR con 3 métodos:
    1. Histórico    — percentiles de rendimientos reales
    2. Paramétrico  — asume distribución normal
    3. Monte Carlo  — 10,000 simulaciones aleatorias

    También incluye backtesting de Kupiec.
    """
    # 1. Descargar precios
    datos_dict = descargar_multiples_precios(tickers, fecha_inicio, fecha_fin)

    precios_dict = {}
    for ticker in tickers:
        if datos_dict[ticker] is not None:
            precios_dict[ticker] = datos_dict[ticker].set_index("fecha")["cierre"]

    if not precios_dict:
        raise ValueError("No se pudieron obtener datos de ningún activo")

    df_precios      = pd.DataFrame(precios_dict).dropna()
    rendimientos    = np.log(df_precios / df_precios.shift(1)).dropna()
    pesos_array     = np.array(pesos)
    rend_port       = rendimientos.values @ pesos_array
    alpha           = 1 - nivel_confianza

    # ── Método 1: Histórico ───────────────────────────────────────────────────
    var_hist  = float(np.percentile(rend_port, alpha * 100))
    cvar_hist = float(rend_port[rend_port <= var_hist].mean())

    # ── Método 2: Paramétrico ─────────────────────────────────────────────────
    media     = float(rend_port.mean())
    std       = float(rend_port.std())
    z         = float(stats.norm.ppf(alpha))
    var_param = float(media + z * std)
    cvar_param= float(media - std * stats.norm.pdf(z) / alpha)

    # ── Método 3: Monte Carlo ─────────────────────────────────────────────────
    np.random.seed(42)
    sim       = np.random.normal(media, std, 10_000)
    var_mc    = float(np.percentile(sim, alpha * 100))
    cvar_mc   = float(sim[sim <= var_mc].mean())

    # ── Monetario ─────────────────────────────────────────────────────────────
    def a_usd(v):
        return round(float(v) * valor_portafolio, 2)

    # ── Backtesting Kupiec ────────────────────────────────────────────────────
    excedencias      = int(np.sum(rend_port < var_hist))
    total_obs        = int(len(rend_port))
    tasa_real        = round(excedencias / total_obs, 4)
    tasa_esperada    = round(alpha, 4)
    modelo_adecuado  = bool(abs(tasa_real - tasa_esperada) < 0.02)

    resultado = {
        "portafolio": {
            "tickers":          tickers,
            "pesos":            pesos,
            "valor_portafolio": valor_portafolio,
            "moneda":           "USD",
        },
        "parametros": {
            "nivel_confianza": nivel_confianza,
            "fecha_inicio":    fecha_inicio,
            "total_obs":       total_obs,
        },
        "estadisticas_portafolio": {
            "rendimiento_medio_diario": round(media, 6),
            "volatilidad_diaria":       round(std, 6),
            "rendimiento_anual":        round(media * 252, 4),
            "volatilidad_anual":        round(std * np.sqrt(252), 4),
            "sharpe_ratio":             round((media * 252) / (std * np.sqrt(252)), 4) if std > 0 else None,
        },
        "var_historico": {
            "var_decimal":     round(var_hist, 6),
            "var_porcentaje":  f"{abs(var_hist)*100:.2f}%",
            "var_monetario":   a_usd(var_hist),
            "cvar_decimal":    round(cvar_hist, 6),
            "cvar_porcentaje": f"{abs(cvar_hist)*100:.2f}%",
            "cvar_monetario":  a_usd(cvar_hist),
            "interpretacion":  (
                f"Con {nivel_confianza*100:.0f}% de confianza, la pérdida máxima diaria "
                f"no excederá ${abs(a_usd(var_hist)):,.2f} USD "
                f"({abs(var_hist)*100:.2f}% del portafolio)"
            ),
        },
        "var_parametrico": {
            "var_decimal":     round(var_param, 6),
            "var_porcentaje":  f"{abs(var_param)*100:.2f}%",
            "var_monetario":   a_usd(var_param),
            "cvar_decimal":    round(cvar_param, 6),
            "cvar_porcentaje": f"{abs(cvar_param)*100:.2f}%",
            "cvar_monetario":  a_usd(cvar_param),
            "supuesto":        "Distribución normal de rendimientos",
        },
        "var_monte_carlo": {
            "var_decimal":     round(var_mc, 6),
            "var_porcentaje":  f"{abs(var_mc)*100:.2f}%",
            "var_monetario":   a_usd(var_mc),
            "cvar_decimal":    round(cvar_mc, 6),
            "cvar_porcentaje": f"{abs(cvar_mc)*100:.2f}%",
            "cvar_monetario":  a_usd(cvar_mc),
            "simulaciones":    10_000,
        },
        "backtesting_kupiec": {
            "excedencias_observadas":    excedencias,
            "total_observaciones":       total_obs,
            "tasa_excedencias_real":     tasa_real,
            "tasa_excedencias_esperada": tasa_esperada,
            "modelo_adecuado":           modelo_adecuado,
            "interpretacion": (
                "Modelo adecuado — excedencias reales coinciden con las esperadas"
                if modelo_adecuado
                else "Modelo puede subestimar el riesgo — revisar supuestos"
            ),
        },
    }

    return _limpiar_dict(resultado)
