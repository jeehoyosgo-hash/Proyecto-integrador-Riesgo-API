import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Optional, List
from services.datos import descargar_precios, descargar_multiples_precios


def _limpiar(v):
    """Convierte tipos numpy a tipos nativos de Python para JSON."""
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


def _limpiar_dict(d):
    """Limpia recursivamente un diccionario o lista."""
    if isinstance(d, dict):
        return {k: _limpiar_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_limpiar_dict(i) for i in d]
    return _limpiar(d)


# ─────────────────────────────────────────────
# CAPM
# ─────────────────────────────────────────────

def calcular_capm(
    tickers: List[str],
    benchmark: str = "^GSPC",        # S&P 500
    tasa_libre_riesgo: float = 0.0525,  # 5.25% anual (T-Bills 2024)
    fecha_inicio: str = "2022-01-01",
    fecha_fin: Optional[str] = None,
) -> dict:
    """
    Calcula el modelo CAPM para cada activo del portafolio.

    CAPM: E(Ri) = Rf + Beta_i × (E(Rm) - Rf)

    Donde:
      E(Ri) = rendimiento esperado del activo i
      Rf    = tasa libre de riesgo
      Beta  = sensibilidad del activo al mercado
      E(Rm) = rendimiento esperado del mercado

    Beta > 1 → más volátil que el mercado (más riesgo, más retorno esperado)
    Beta < 1 → menos volátil que el mercado (menos riesgo)
    Beta = 1 → se mueve igual que el mercado
    Beta < 0 → se mueve en dirección opuesta al mercado
    """
    # 1. Descargar precios de todos los activos + benchmark
    todos_tickers = tickers + [benchmark]
    datos         = descargar_multiples_precios(todos_tickers, fecha_inicio, fecha_fin)

    # 2. Construir DataFrame de rendimientos
    rend_dict = {}
    for ticker in todos_tickers:
        if datos[ticker] is not None:
            precios = datos[ticker].set_index("fecha")["cierre"]
            rend_dict[ticker] = np.log(precios / precios.shift(1)).dropna()

    if benchmark not in rend_dict:
        raise ValueError(f"No se pudieron obtener datos del benchmark {benchmark}")

    rend_mercado = rend_dict[benchmark]

    # 3. Calcular Beta y CAPM para cada activo
    rf_diario  = tasa_libre_riesgo / 252  # convertir tasa anual a diaria
    prima_riesgo_mercado_anual = float(rend_mercado.mean() * 252) - tasa_libre_riesgo

    resultados_activos = {}

    for ticker in tickers:
        if ticker not in rend_dict:
            continue

        rend_activo = rend_dict[ticker]

        # Alinear fechas entre activo y benchmark
        df_alineado  = pd.DataFrame({
            "activo":   rend_activo,
            "mercado":  rend_mercado,
        }).dropna()

        if len(df_alineado) < 30:
            continue

        # Calcular Beta mediante regresión lineal
        # Beta = Cov(Ri, Rm) / Var(Rm)
        covarianza   = np.cov(df_alineado["activo"], df_alineado["mercado"])
        beta         = float(covarianza[0, 1] / covarianza[1, 1])

        # Calcular Alpha (intercepto de la regresión)
        # Alpha > 0 → el activo genera retorno por encima de lo esperado por CAPM
        rend_medio_activo  = float(df_alineado["activo"].mean())
        rend_medio_mercado = float(df_alineado["mercado"].mean())
        alpha_diario       = rend_medio_activo - beta * rend_medio_mercado
        alpha_anual        = alpha_diario * 252

        # Rendimiento esperado por CAPM (anualizado)
        rendimiento_esperado = tasa_libre_riesgo + beta * prima_riesgo_mercado_anual

        # R² de la regresión (qué tanto explica el mercado la variación del activo)
        correlacion = float(np.corrcoef(df_alineado["activo"], df_alineado["mercado"])[0, 1])
        r_cuadrado  = correlacion ** 2

        # Riesgo total vs riesgo sistemático
        vol_activo    = float(df_alineado["activo"].std() * np.sqrt(252))
        vol_mercado   = float(df_alineado["mercado"].std() * np.sqrt(252))
        riesgo_sist   = beta * vol_mercado      # riesgo que no se puede diversificar
        riesgo_idios  = float(np.sqrt(max(vol_activo**2 - riesgo_sist**2, 0)))

        resultados_activos[ticker] = {
            "beta":                    round(beta, 4),
            "alpha_anual":             round(alpha_anual, 4),
            "rendimiento_esperado_capm": round(rendimiento_esperado, 4),
            "rendimiento_esperado_pct":  f"{rendimiento_esperado*100:.2f}%",
            "r_cuadrado":              round(r_cuadrado, 4),
            "correlacion_mercado":     round(correlacion, 4),
            "volatilidad_anual":       round(vol_activo, 4),
            "riesgo_sistematico":      round(riesgo_sist, 4),
            "riesgo_idiosincratico":   round(riesgo_idios, 4),
            "observaciones":           len(df_alineado),
            "interpretacion_beta":     _interpretar_beta(beta),
            "interpretacion_alpha":    (
                f"Alpha positivo ({alpha_anual*100:.2f}%) — genera valor por encima del CAPM"
                if alpha_anual > 0
                else f"Alpha negativo ({alpha_anual*100:.2f}%) — por debajo de lo esperado por CAPM"
            ),
        }

    return _limpiar_dict({
        "benchmark":                   benchmark,
        "tasa_libre_riesgo_anual":     tasa_libre_riesgo,
        "tasa_libre_riesgo_pct":       f"{tasa_libre_riesgo*100:.2f}%",
        "rendimiento_mercado_anual":   round(float(rend_mercado.mean() * 252), 4),
        "prima_riesgo_mercado":        round(prima_riesgo_mercado_anual, 4),
        "prima_riesgo_mercado_pct":    f"{prima_riesgo_mercado_anual*100:.2f}%",
        "activos":                     resultados_activos,
    })


def _interpretar_beta(beta: float) -> str:
    if beta > 1.5:
        return f"Beta muy alta ({beta:.2f}) — muy sensible al mercado, alto riesgo sistemático"
    elif beta > 1.0:
        return f"Beta alta ({beta:.2f}) — más volátil que el mercado"
    elif beta > 0.5:
        return f"Beta moderada ({beta:.2f}) — menos volátil que el mercado"
    elif beta > 0:
        return f"Beta baja ({beta:.2f}) — muy poco sensible al mercado"
    else:
        return f"Beta negativa ({beta:.2f}) — se mueve en dirección opuesta al mercado"


# ─────────────────────────────────────────────
# FRONTERA EFICIENTE DE MARKOWITZ
# ─────────────────────────────────────────────

def calcular_frontera_eficiente(
    tickers: List[str],
    fecha_inicio: str = "2022-01-01",
    fecha_fin: Optional[str] = None,
    tasa_libre_riesgo: float = 0.0525,
    n_portafolios: int = 500,
) -> dict:
    """
    Construye la frontera eficiente de Markowitz.

    La teoría de Markowitz dice que para un nivel de riesgo dado,
    existe un portafolio que maximiza el retorno esperado.
    El conjunto de todos esos portafolios forma la 'frontera eficiente'.

    Portafolios especiales:
    - Mínima varianza: el de menor riesgo posible
    - Máximo Sharpe:   el mejor balance riesgo/retorno
    """
    # 1. Descargar precios y calcular rendimientos
    datos    = descargar_multiples_precios(tickers, fecha_inicio, fecha_fin)
    rend_dict = {}
    for ticker in tickers:
        if datos[ticker] is not None:
            precios = datos[ticker].set_index("fecha")["cierre"]
            rend_dict[ticker] = np.log(precios / precios.shift(1)).dropna()

    df_rend = pd.DataFrame(rend_dict).dropna()
    n       = len(tickers)

    # 2. Parámetros estadísticos anualizados
    medias   = df_rend.mean() * 252               # retornos esperados anuales
    cov_mat  = df_rend.cov() * 252                # matriz de covarianza anual
    rf_anual = tasa_libre_riesgo

    # 3. Simular n_portafolios aleatorios
    np.random.seed(42)
    port_retornos    = []
    port_volatilidades = []
    port_sharpes     = []
    port_pesos_lista = []

    for _ in range(n_portafolios):
        # Generar pesos aleatorios que sumen 1
        w = np.random.dirichlet(np.ones(n))

        ret = float(w @ medias.values)
        vol = float(np.sqrt(w @ cov_mat.values @ w))
        sharpe = (ret - rf_anual) / vol if vol > 0 else 0

        port_retornos.append(round(ret, 6))
        port_volatilidades.append(round(vol, 6))
        port_sharpes.append(round(sharpe, 4))
        port_pesos_lista.append([round(float(p), 4) for p in w])

    # 4. Portafolio de MÍNIMA VARIANZA
    def varianza_port(w):
        return float(w @ cov_mat.values @ w)

    restricciones = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    limites       = tuple((0, 1) for _ in range(n))
    w0            = np.ones(n) / n  # pesos iniciales iguales

    res_min_var = minimize(
        varianza_port,
        w0,
        method="SLSQP",
        bounds=limites,
        constraints=restricciones,
    )

    w_min_var   = res_min_var.x
    ret_min_var = float(w_min_var @ medias.values)
    vol_min_var = float(np.sqrt(w_min_var @ cov_mat.values @ w_min_var))
    sharpe_min_var = (ret_min_var - rf_anual) / vol_min_var if vol_min_var > 0 else 0

    # 5. Portafolio de MÁXIMO SHARPE RATIO
    def neg_sharpe(w):
        ret = float(w @ medias.values)
        vol = float(np.sqrt(w @ cov_mat.values @ w))
        return -(ret - rf_anual) / vol if vol > 0 else 0

    res_max_sharpe = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=limites,
        constraints=restricciones,
    )

    w_max_sharpe   = res_max_sharpe.x
    ret_max_sharpe = float(w_max_sharpe @ medias.values)
    vol_max_sharpe = float(np.sqrt(w_max_sharpe @ cov_mat.values @ w_max_sharpe))
    sharpe_max     = (ret_max_sharpe - rf_anual) / vol_max_sharpe if vol_max_sharpe > 0 else 0

    # 6. Portafolio IGUALMENTE PONDERADO (benchmark simple)
    w_igual   = np.ones(n) / n
    ret_igual = float(w_igual @ medias.values)
    vol_igual = float(np.sqrt(w_igual @ cov_mat.values @ w_igual))
    sharpe_igual = (ret_igual - rf_anual) / vol_igual if vol_igual > 0 else 0

    # 7. Construir frontera eficiente (puntos óptimos)
    retornos_objetivo = np.linspace(
        min(port_retornos),
        max(port_retornos),
        50,
    )
    frontera = []
    for ret_obj in retornos_objetivo:
        restricciones_frontera = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, r=ret_obj: float(w @ medias.values) - r},
        ]
        res = minimize(
            varianza_port,
            w0,
            method="SLSQP",
            bounds=limites,
            constraints=restricciones_frontera,
        )
        if res.success:
            v = float(np.sqrt(res.fun))
            frontera.append({
                "retorno":     round(float(ret_obj), 4),
                "volatilidad": round(v, 4),
                "sharpe":      round((float(ret_obj) - rf_anual) / v, 4) if v > 0 else 0,
            })

    return _limpiar_dict({
        "tickers":             tickers,
        "tasa_libre_riesgo":   tasa_libre_riesgo,
        "portafolio_min_varianza": {
            "pesos":       {t: round(float(w), 4) for t, w in zip(tickers, w_min_var)},
            "retorno_anual":    round(ret_min_var, 4),
            "retorno_pct":      f"{ret_min_var*100:.2f}%",
            "volatilidad_anual":round(vol_min_var, 4),
            "volatilidad_pct":  f"{vol_min_var*100:.2f}%",
            "sharpe_ratio":     round(sharpe_min_var, 4),
            "interpretacion":   "Portafolio con el menor riesgo posible dados estos activos",
        },
        "portafolio_max_sharpe": {
            "pesos":       {t: round(float(w), 4) for t, w in zip(tickers, w_max_sharpe)},
            "retorno_anual":    round(ret_max_sharpe, 4),
            "retorno_pct":      f"{ret_max_sharpe*100:.2f}%",
            "volatilidad_anual":round(vol_max_sharpe, 4),
            "volatilidad_pct":  f"{vol_max_sharpe*100:.2f}%",
            "sharpe_ratio":     round(sharpe_max, 4),
            "interpretacion":   "Portafolio con el mejor balance riesgo/retorno (máximo Sharpe Ratio)",
        },
        "portafolio_igual_ponderado": {
            "pesos":       {t: round(float(w), 4) for t, w in zip(tickers, w_igual)},
            "retorno_anual":    round(ret_igual, 4),
            "retorno_pct":      f"{ret_igual*100:.2f}%",
            "volatilidad_anual":round(vol_igual, 4),
            "volatilidad_pct":  f"{vol_igual*100:.2f}%",
            "sharpe_ratio":     round(sharpe_igual, 4),
            "interpretacion":   "Portafolio con pesos iguales para todos los activos",
        },
        "frontera_eficiente":  frontera,
        "simulacion": {
            "n_portafolios":    n_portafolios,
            "retornos":         port_retornos,
            "volatilidades":    port_volatilidades,
            "sharpes":          port_sharpes,
            "pesos":            port_pesos_lista,
        },
    })
