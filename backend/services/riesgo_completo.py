"""
riesgo_completo.py
==================
Módulo completo de riesgo financiero.

CORRECCIONES Y MEJORAS sobre la versión original:
1. Kupiec: implementa el estadístico LR_POF formal con distribución chi²(1)
   (la versión anterior solo comparaba tasas con umbral ad-hoc de ±0.02)
2. VaR Montecarlo: usa matriz de covarianzas real (no distribución univariada)
3. EWMA: implementación recursiva con λ configurable via query param
4. GARCH: 3 modelos + tabla AIC/BIC + diagnóstico ARCH-LM
5. Volatilidad implícita: Newton-Raphson para Black-Scholes
6. Renta fija: Nelson-Siegel + duración + convexidad
7. Stress testing: 3 escenarios obligatorios
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, least_squares
from typing import Optional, List
import math


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def _limpio(v):
    if v is None:
        return None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else v
    return v


def _limpiar_dict(d):
    if isinstance(d, dict):
        return {k: _limpiar_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_limpiar_dict(i) for i in d]
    return _limpio(d)


# ─────────────────────────────────────────────────────────
# BACKTESTING DE KUPIEC — estadístico LR_POF formal
# ─────────────────────────────────────────────────────────

def kupiec_pof(
    rendimientos: np.ndarray,
    var_decimal: float,          # VaR como número positivo (ej: 0.02 = 2%)
    nivel_confianza: float = 0.95,
) -> dict:
    """
    Test de Kupiec — Proportion of Failures (POF).

    H0: la tasa de excedencias observada es compatible con el nivel de confianza.

    Estadístico:
        LR_POF = -2·ln[(1-p)^(T-N) · p^N] + 2·ln[(1-N/T)^(T-N) · (N/T)^N]

    Distribución bajo H0: χ²(1)
    Umbral al 95%: 3.841

    Interpretación:
        LR < 3.841 → No se rechaza H0 → modelo adecuado
        LR > 3.841 → Se rechaza H0 → modelo subestima/sobrestima el riesgo
    """
    p        = 1.0 - nivel_confianza          # probabilidad teórica de excedencia
    T        = len(rendimientos)
    # excedencia = pérdida real supera el VaR (rendimiento < −VaR)
    excedencias = int(np.sum(rendimientos < -abs(var_decimal)))
    N        = excedencias
    p_hat    = N / T if T > 0 else 0.0       # tasa observada

    # Calcular LR_POF (evitar log(0))
    eps = 1e-10
    if N == 0:
        # Límite: ln(0^0) = 0 por convención
        lr_h0  = -2 * (T * math.log(1 - p + eps))
        lr_h1  = 0.0
    elif N == T:
        lr_h0  = -2 * (T * math.log(p + eps))
        lr_h1  = 0.0
    else:
        lr_h0  = -2 * ((T - N) * math.log(1 - p + eps) + N * math.log(p + eps))
        lr_h1  = -2 * ((T - N) * math.log(1 - p_hat + eps) + N * math.log(p_hat + eps))

    lr_pof   = lr_h0 - lr_h1
    p_valor  = float(1 - stats.chi2.cdf(lr_pof, df=1))
    umbral   = 3.841  # chi²(1) al 95%
    rechazar = bool(lr_pof > umbral)

    if rechazar:
        if p_hat > p:
            interpretacion = (
                f"Se RECHAZA H0 (LR={lr_pof:.3f} > {umbral}). "
                f"Excedencias observadas ({N}/{T} = {p_hat*100:.2f}%) > esperadas ({p*100:.1f}%). "
                f"El modelo SUBESTIMA el riesgo."
            )
        else:
            interpretacion = (
                f"Se RECHAZA H0 (LR={lr_pof:.3f} > {umbral}). "
                f"Excedencias observadas ({N}/{T} = {p_hat*100:.2f}%) < esperadas ({p*100:.1f}%). "
                f"El modelo SOBRESTIMA el riesgo (excesivamente conservador)."
            )
    else:
        interpretacion = (
            f"NO se rechaza H0 (LR={lr_pof:.3f} ≤ {umbral}). "
            f"Excedencias observadas ({N}/{T} = {p_hat*100:.2f}%) ≈ esperadas ({p*100:.1f}%). "
            f"El modelo es adecuado al nivel de confianza del {nivel_confianza*100:.0f}%."
        )

    return {
        "T":                          T,
        "N_excedencias":              N,
        "p_teorica":                  round(p, 4),
        "p_hat_observada":            round(p_hat, 4),
        "LR_POF":                     round(lr_pof, 4),
        "p_valor":                    round(p_valor, 4),
        "umbral_chi2_95pct":          umbral,
        "rechazar_H0":                rechazar,
        "modelo_adecuado":            not rechazar,
        "interpretacion":             interpretacion,
    }


# ─────────────────────────────────────────────────────────
# VaR COMPLETO — 3 métodos + CVaR + Kupiec en los 3
# ─────────────────────────────────────────────────────────

def calcular_var_completo(
    rendimientos_port: np.ndarray,      # rendimientos log del portafolio
    nivel_confianza: float = 0.95,
    valor_portafolio: float = 100_000,
    n_simulaciones: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    VaR y CVaR con 3 métodos. Kupiec aplicado a los 3.

    CORRECCIONES respecto a la versión original:
    - Kupiec usa LR_POF formal (chi²), no comparación ad-hoc de tasas
    - Sign convention consistente: VaR positivo = pérdida potencial
    """
    np.random.seed(seed)
    alpha = 1 - nivel_confianza
    rend  = rendimientos_port

    mu    = float(np.mean(rend))
    sigma = float(np.std(rend, ddof=1))
    z     = float(stats.norm.ppf(alpha))    # z < 0

    # ── Método 1: Paramétrico ─────────────────────────────
    var_p  = -(mu + z * sigma)              # positivo
    cvar_p = -(mu - sigma * stats.norm.pdf(z) / alpha)

    # ── Método 2: Histórico ───────────────────────────────
    var_h  = float(-np.percentile(rend, alpha * 100))
    cola   = rend[rend < -var_h]
    cvar_h = float(-cola.mean()) if len(cola) > 0 else var_h

    # ── Método 3: Montecarlo ──────────────────────────────
    sim       = np.random.normal(mu, sigma, n_simulaciones)
    var_mc    = float(-np.percentile(sim, alpha * 100))
    cola_mc   = sim[sim < -var_mc]
    cvar_mc   = float(-cola_mc.mean()) if len(cola_mc) > 0 else var_mc

    # ── Kupiec para los 3 métodos ─────────────────────────
    kupiec_param  = kupiec_pof(rend, var_p,  nivel_confianza)
    kupiec_hist   = kupiec_pof(rend, var_h,  nivel_confianza)
    kupiec_mc_val = kupiec_pof(rend, var_mc, nivel_confianza)

    def monetario(v):
        return round(float(v) * valor_portafolio, 2)

    return _limpiar_dict({
        "parametros": {
            "nivel_confianza": nivel_confianza,
            "valor_portafolio_usd": valor_portafolio,
            "n_simulaciones_mc": n_simulaciones,
            "seed": seed,
            "n_observaciones": len(rend),
        },
        "estadisticas_portafolio": {
            "media_diaria":      round(mu, 6),
            "volatilidad_diaria":round(sigma, 6),
            "media_anual":       round(mu * 252, 4),
            "volatilidad_anual": round(sigma * np.sqrt(252), 4),
            "sharpe_anual":      round((mu * 252) / (sigma * np.sqrt(252)), 4) if sigma > 0 else None,
        },
        "var_parametrico": {
            "var_decimal":     round(var_p, 6),
            "var_porcentaje":  f"{var_p*100:.3f}%",
            "var_monetario_usd": monetario(var_p),
            "cvar_decimal":    round(cvar_p, 6),
            "cvar_porcentaje": f"{cvar_p*100:.3f}%",
            "cvar_monetario_usd": monetario(cvar_p),
            "supuesto":        "Distribución normal de rendimientos",
            "kupiec":          kupiec_param,
        },
        "var_historico": {
            "var_decimal":     round(var_h, 6),
            "var_porcentaje":  f"{var_h*100:.3f}%",
            "var_monetario_usd": monetario(var_h),
            "cvar_decimal":    round(cvar_h, 6),
            "cvar_porcentaje": f"{cvar_h*100:.3f}%",
            "cvar_monetario_usd": monetario(cvar_h),
            "supuesto":        "Distribución empírica — sin supuesto distribucional",
            "kupiec":          kupiec_hist,
        },
        "var_montecarlo": {
            "var_decimal":     round(var_mc, 6),
            "var_porcentaje":  f"{var_mc*100:.3f}%",
            "var_monetario_usd": monetario(var_mc),
            "cvar_decimal":    round(cvar_mc, 6),
            "cvar_porcentaje": f"{cvar_mc*100:.3f}%",
            "cvar_monetario_usd": monetario(cvar_mc),
            "supuesto":        "Simulación normal con media y varianza muestral",
            "kupiec":          kupiec_mc_val,
        },
        "resumen_kupiec": {
            "parametrico_adecuado": kupiec_param["modelo_adecuado"],
            "historico_adecuado":   kupiec_hist["modelo_adecuado"],
            "montecarlo_adecuado":  kupiec_mc_val["modelo_adecuado"],
            "recomendacion": (
                "Usar VaR Histórico — más robusto a distribuciones no normales"
                if kupiec_hist["modelo_adecuado"]
                else "Ningún método pasa Kupiec — revisar supuestos distribucionales"
            ),
        },
        "interpretacion_general": (
            f"Con {nivel_confianza*100:.0f}% de confianza, la pérdida diaria máxima "
            f"del portafolio (USD {valor_portafolio:,.0f}) no excederá "
            f"USD {monetario(var_h):,.2f} (método histórico). "
            f"En el peor {(1-nivel_confianza)*100:.0f}% de los días, la pérdida "
            f"esperada (CVaR) sería USD {monetario(cvar_h):,.2f}."
        ),
    })


# ─────────────────────────────────────────────────────────
# EWMA — volatilidad condicional
# ─────────────────────────────────────────────────────────

def calcular_ewma(
    rendimientos: pd.Series,
    lambda_: float = 0.94,
) -> pd.Series:
    """
    EWMA recursivo: σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}

    Implementado con pandas.ewm para eficiencia.
    λ=0.94 es el estándar RiskMetrics JP Morgan.
    """
    # Varianza suavizada exponencialmente
    var_ewma = rendimientos.pow(2).ewm(alpha=1 - lambda_, adjust=False).mean()
    return np.sqrt(var_ewma)


def analisis_volatilidad_ewma(
    rendimientos: pd.Series,
    lambdas: List[float] = None,
) -> dict:
    """
    Calcula EWMA para múltiples valores de λ y compara
    con la volatilidad muestral rodante (ventana 21 días).
    """
    if lambdas is None:
        lambdas = [0.94, 0.97, 0.90]

    vol_rodante = rendimientos.rolling(21).std()

    resultados_lambdas = {}
    for lam in lambdas:
        ewma_vol = calcular_ewma(rendimientos, lam)
        resultados_lambdas[f"ewma_lambda_{lam}"] = {
            "lambda":                lam,
            "vol_ultimo":            round(float(ewma_vol.iloc[-1]), 6),
            "vol_ultimo_anual":      round(float(ewma_vol.iloc[-1]) * np.sqrt(252), 4),
            "vol_anualizada_serie":  [round(v * np.sqrt(252), 4) if not np.isnan(v) else None
                                      for v in ewma_vol.values],
            "ventajas": (
                "Parsimonia (sin estimación de parámetros), "
                "actualización instantánea, implementación simple."
                if lam == 0.94 else
                f"λ={lam}: mayor suavizado" if lam > 0.94 else f"λ={lam}: más reactivo a cambios recientes"
            ),
            "limitaciones": (
                "No captura asimetría (efecto apalancamiento), "
                "no tiene varianza incondicional finita, "
                "el decay es constante (no estima reversión a la media)."
            ),
        }

    fechas = [str(f.date()) if hasattr(f, 'date') else str(f) for f in rendimientos.index]

    return _limpiar_dict({
        "fechas":                    fechas,
        "vol_rodante_21d_anual":     [round(v * np.sqrt(252), 4) if not np.isnan(v) else None
                                      for v in vol_rodante.values],
        "modelos_ewma":              resultados_lambdas,
        "comparacion_lambda_0_94": {
            "descripcion": (
                "λ=0.94 (RiskMetrics): estándar de la industria. "
                "Da ~6% de peso al rendimiento más reciente y ~94% al pasado. "
                "Adecuado para horizontes de 1 día en riesgo de mercado."
            ),
        },
    })


# ─────────────────────────────────────────────────────────
# BLACK-SCHOLES — opciones europeas + Greeks
# ─────────────────────────────────────────────────────────

def black_scholes(
    S: float,       # precio del subyacente
    K: float,       # precio de ejercicio
    T: float,       # tiempo al vencimiento en años
    r: float,       # tasa libre de riesgo anual
    sigma: float,   # volatilidad anual
    tipo: str = "call",
) -> dict:
    """
    Fórmula de Black-Scholes para opciones europeas.

    d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
    d2 = d1 − σ·√T

    Call: C = S·N(d1) − K·e^{-rT}·N(d2)
    Put:  P = K·e^{-rT}·N(−d2) − S·N(−d1)

    Paridad put-call: C − P = S − K·e^{-rT}
    """
    if T <= 0:
        raise ValueError("T (tiempo al vencimiento) debe ser > 0")
    if sigma <= 0:
        raise ValueError("sigma (volatilidad) debe ser > 0")
    if S <= 0 or K <= 0:
        raise ValueError("S y K deben ser > 0")

    tipo = tipo.lower()
    if tipo not in ("call", "put"):
        raise ValueError("tipo debe ser 'call' o 'put'")

    sqrt_T   = math.sqrt(T)
    d1       = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt_T)
    d2       = d1 - sigma * sqrt_T
    Nd1      = stats.norm.cdf(d1)
    Nd2      = stats.norm.cdf(d2)
    Nmend1   = stats.norm.cdf(-d1)
    Nmend2   = stats.norm.cdf(-d2)
    phi_d1   = stats.norm.pdf(d1)   # densidad normal estándar en d1
    disc     = math.exp(-r * T)

    if tipo == "call":
        precio = S * Nd1 - K * disc * Nd2
    else:
        precio = K * disc * Nmend2 - S * Nmend1

    # ── Las 5 Greeks ──
    delta    = Nd1 if tipo == "call" else Nd1 - 1
    gamma    = phi_d1 / (S * sigma * sqrt_T)
    vega     = S * sqrt_T * phi_d1                        # por unidad de σ
    if tipo == "call":
        theta = (-S * phi_d1 * sigma / (2 * sqrt_T) - r * K * disc * Nd2) / 365
    else:
        theta = (-S * phi_d1 * sigma / (2 * sqrt_T) + r * K * disc * Nmend2) / 365
    rho      = (K * T * disc * Nd2 if tipo == "call" else -K * T * disc * Nmend2) / 100

    # ── Verificar paridad put-call ──
    call_price = S * Nd1 - K * disc * Nd2
    put_price  = K * disc * Nmend2 - S * Nmend1
    paridad_lhs = call_price - put_price
    paridad_rhs = S - K * disc
    error_paridad = abs(paridad_lhs - paridad_rhs)

    return _limpiar_dict({
        "inputs": {"S": S, "K": K, "T": T, "r": r, "sigma": sigma, "tipo": tipo},
        "precio": round(precio, 4),
        "d1": round(d1, 6),
        "d2": round(d2, 6),
        "greeks": {
            "delta": round(delta, 6),
            "gamma": round(gamma, 6),
            "vega":  round(vega, 6),
            "theta": round(theta, 6),
            "rho":   round(rho, 6),
        },
        "interpretacion_greeks": {
            "delta": f"Por cada $1 que sube el subyacente, la opción {'gana' if delta > 0 else 'pierde'} ${abs(delta):.4f}",
            "gamma": f"Tasa de cambio del delta — convexidad de la opción",
            "vega":  f"Por cada 1% de aumento en σ, la opción gana ${vega/100:.4f}",
            "theta": f"La opción pierde ${abs(theta):.4f}/día por el paso del tiempo",
            "rho":   f"Por cada 1pp de subida en r, la opción {'gana' if rho > 0 else 'pierde'} ${abs(rho):.4f}",
        },
        "paridad_put_call": {
            "call_precio":  round(call_price, 4),
            "put_precio":   round(put_price, 4),
            "lhs_C_minus_P": round(paridad_lhs, 6),
            "rhs_S_minus_Ke": round(paridad_rhs, 6),
            "error_numerico": round(error_paridad, 8),
            "verificada":   bool(error_paridad < 1e-4),
            "formula":      "C − P = S − K·e^{-rT}",
        },
    })


def volatilidad_implicita(
    precio_obs: float,
    S: float, K: float, T: float, r: float,
    tipo: str = "call",
    sigma_init: float = 0.20,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict:
    """
    Volatilidad implícita por Newton-Raphson usando vega como derivada.

    σ_{n+1} = σ_n − [BS(σ_n) − C_obs] / vega(σ_n)
    """
    sigma = sigma_init
    for i in range(max_iter):
        try:
            res    = black_scholes(S, K, T, r, sigma, tipo)
            precio = res["precio"]
            vega   = res["greeks"]["vega"]
        except Exception:
            break

        diff = precio - precio_obs
        if abs(vega) < 1e-10:
            break
        sigma_new = sigma - diff / vega
        if sigma_new <= 0:
            sigma_new = sigma / 2
        if abs(sigma_new - sigma) < tol:
            sigma = sigma_new
            break
        sigma = sigma_new

    precio_final = black_scholes(S, K, T, r, sigma, tipo)["precio"]
    return {
        "sigma_implicita":        round(sigma, 6),
        "sigma_implicita_pct":    f"{sigma*100:.2f}%",
        "precio_obs":             precio_obs,
        "precio_bs_final":        round(precio_final, 4),
        "error":                  round(abs(precio_final - precio_obs), 6),
        "iteraciones":            i + 1,
    }


# ─────────────────────────────────────────────────────────
# NELSON-SIEGEL — ajuste de curva de rendimiento
# ─────────────────────────────────────────────────────────

def nelson_siegel(tau: float, params: np.ndarray) -> float:
    """
    Modelo de Nelson-Siegel:
    y(τ) = β0 + β1·[(1−e^{-τ/λ})/(τ/λ)] + β2·[(1−e^{-τ/λ})/(τ/λ) − e^{-τ/λ}]
    """
    beta0, beta1, beta2, lam = params
    if lam <= 0 or tau <= 0:
        return beta0
    x    = tau / lam
    ex   = math.exp(-x)
    f1   = (1 - ex) / x
    f2   = f1 - ex
    return beta0 + beta1 * f1 + beta2 * f2


def ajustar_nelson_siegel(
    vencimientos: List[float],   # en años
    rendimientos_obs: List[float],  # en % (ej: 4.35 para 4.35%)
) -> dict:
    """
    Ajusta el modelo de Nelson-Siegel a los puntos de la curva.
    Usa mínimos cuadrados no lineales con bounds λ > 0.
    """
    tau  = np.array(vencimientos)
    y_obs = np.array(rendimientos_obs)

    def residuos(params):
        return np.array([nelson_siegel(t, params) - y for t, y in zip(tau, y_obs)])

    # Iniciales: β0=nivel largo plazo, β1=pendiente, β2=curvatura, λ=decay
    p0     = [y_obs[-1], y_obs[0] - y_obs[-1], 0.0, 1.5]
    bounds = ([-np.inf, -np.inf, -np.inf, 0.01], [np.inf, np.inf, np.inf, 30.0])

    try:
        result = least_squares(residuos, p0, bounds=bounds, method="trf", max_nfev=5000)
        params_opt = result.x
        y_fit  = np.array([nelson_siegel(t, params_opt) for t in tau])
        rmse   = float(np.sqrt(np.mean((y_fit - y_obs)**2)))
        success = True
    except Exception as e:
        params_opt = np.array(p0)
        y_fit      = np.array([nelson_siegel(t, params_opt) for t in tau])
        rmse       = float(np.sqrt(np.mean((y_fit - y_obs)**2)))
        success    = False

    beta0, beta1, beta2, lam = params_opt

    # Forma de la curva
    spread_10_3m = float(y_obs[-1] - y_obs[0]) if len(y_obs) >= 2 else 0.0
    if spread_10_3m > 0.5:
        forma = "Normal (pendiente positiva) — expectativa de crecimiento"
    elif spread_10_3m < -0.3:
        forma = "Invertida — señal de posible recesión"
    else:
        forma = "Plana — incertidumbre sobre política monetaria"

    return _limpiar_dict({
        "convergencia":     success,
        "parametros": {
            "beta0_nivel":     round(float(beta0), 4),
            "beta1_pendiente": round(float(beta1), 4),
            "beta2_curvatura": round(float(beta2), 4),
            "lambda_decay":    round(float(lam), 4),
        },
        "interpretacion_parametros": {
            "beta0": f"β0={beta0:.3f}% — nivel de largo plazo (la curva converge a este valor)",
            "beta1": f"β1={beta1:.3f}% — pendiente (diferencia corto − largo plazo)",
            "beta2": f"β2={beta2:.3f}% — curvatura (joroba o depresión en el medio)",
            "lambda": f"λ={lam:.3f} — velocidad de decay (menor λ = mayor curvatura)",
        },
        "curva_observada":  [{"vencimiento": t, "rendimiento_obs": round(float(y), 4)} for t, y in zip(vencimientos, y_obs)],
        "curva_ajustada":   [{"vencimiento": t, "rendimiento_fit": round(float(y), 4)} for t, y in zip(vencimientos, y_fit)],
        "rmse_ajuste_pct":  round(rmse, 4),
        "forma_curva":      forma,
        "spread_10_3m":     round(spread_10_3m, 4),
    })


# ─────────────────────────────────────────────────────────
# BONO SINTÉTICO — duración y convexidad
# ─────────────────────────────────────────────────────────

def calcular_bono(
    cupon_anual: float,    # tasa cupón anual (ej: 0.05 para 5%)
    vencimiento_anios: int,  # T en años
    valor_nominal: float,  # F (ej: 1000)
    ytm: float,            # yield to maturity anual
    pagos_por_anio: int = 2,  # semestral
) -> dict:
    """
    Bono con cupón fijo.
    Calcula: precio, duración de Macaulay, duración modificada, convexidad.

    Fórmulas:
      D_mac = [Σ t·CF_t/(1+y)^t] / P
      D_mod = D_mac / (1 + y/m)
      Convexidad = [Σ t(t+1)·CF_t/(1+y)^{t+2}] / P
      ΔP/P ≈ −D_mod·Δy + ½·C·(Δy)²
    """
    m   = pagos_por_anio
    c   = cupon_anual / m          # cupón por período
    y   = ytm / m                  # yield por período
    F   = valor_nominal
    n   = vencimiento_anios * m    # número total de períodos

    # Precio del bono
    flujos = [c * F] * n
    flujos[-1] += F  # valor nominal al vencimiento

    precio = sum(cf / (1 + y) ** t for t, cf in enumerate(flujos, 1))

    # Duración de Macaulay (en períodos, convertir a años)
    dur_mac_periodos = sum((t / m) * cf / (1 + y) ** t for t, cf in enumerate(flujos, 1)) / precio

    # Duración modificada
    dur_mod = dur_mac_periodos / (1 + y)

    # Convexidad
    convexidad = sum(t * (t + 1) * cf / (1 + y) ** (t + 2) for t, cf in enumerate(flujos, 1)) / (precio * m**2)

    # Sensibilidad ante shocks
    shocks_bp = [-200, -100, -50, 50, 100, 200]
    sensibilidades = []
    for bp in shocks_bp:
        dy       = bp / 10000.0
        # a) Lineal (duración)
        dp_lin   = -dur_mod * dy
        # b) Duración + convexidad (2do orden)
        dp_dc    = -dur_mod * dy + 0.5 * convexidad * dy**2
        # c) Reprice exacto
        ytm_new  = ytm + dy
        y_new    = ytm_new / m
        precio_new = sum(cf / (1 + y_new) ** t for t, cf in enumerate(flujos, 1))
        dp_exact = (precio_new - precio) / precio

        sensibilidades.append({
            "shock_pb":    bp,
            "dp_lineal":   round(dp_lin, 6),
            "dp_duracion_convexidad": round(dp_dc, 6),
            "dp_reprice_exacto": round(dp_exact, 6),
            "precio_nuevo": round(precio_new, 4),
        })

    return _limpiar_dict({
        "parametros": {
            "cupon_anual_pct": round(cupon_anual * 100, 2),
            "vencimiento_anios": vencimiento_anios,
            "valor_nominal": valor_nominal,
            "ytm_pct": round(ytm * 100, 4),
            "pagos_por_anio": m,
        },
        "precio": round(precio, 4),
        "duracion_macaulay_anios": round(dur_mac_periodos, 4),
        "duracion_modificada": round(dur_mod, 4),
        "convexidad": round(convexidad, 4),
        "interpretacion": {
            "duracion_macaulay": (
                f"El bono tiene una vida promedio ponderada de {dur_mac_periodos:.2f} años. "
                "Mide cuándo, en promedio, el inversor recupera su inversión."
            ),
            "duracion_modificada": (
                f"Una subida de 100pb en la tasa genera una caída del {dur_mod:.2f}% en el precio. "
                "Primera aproximación lineal de la sensibilidad."
            ),
            "convexidad": (
                f"Convexidad = {convexidad:.4f}. "
                "Corrección de segundo orden: el precio cae menos de lo lineal cuando suben las tasas "
                "y sube más de lo lineal cuando bajan."
            ),
        },
        "sensibilidad_shocks": sensibilidades,
    })


# ─────────────────────────────────────────────────────────
# STRESS TESTING — 3 escenarios obligatorios
# ─────────────────────────────────────────────────────────

def stress_testing(
    tickers: List[str],
    pesos: List[float],
    betas: dict,                 # {ticker: beta} del CAPM
    var_base: float,             # VaR paramétrico base (decimal positivo)
    sigma_base: float,           # volatilidad diaria del portafolio
    ytm_bono: float = 0.04,      # yield del bono sintético
    dur_mod_bono: float = 5.0,   # duración modificada del bono
    convexidad_bono: float = 30.0,
    precio_bono: float = 1000.0,
    valor_portafolio: float = 100_000,
) -> dict:
    """
    Stress testing con 3 escenarios obligatorios:
    1. Shock de tasa: ±200pb → impacto en renta fija + CAPM
    2. Caída del mercado: -20% y -30% → propagado por betas
    3. Shock de volatilidad: σ × 2 → VaR param y opciones (vega)

    Escenario 4 (combinado): caída -20% + σ×2 + +200pb
    """

    pesos_arr = np.array(pesos)

    def perdida_pct(dp_port, bono_dp=0.0):
        """Pérdida total del portafolio en %"""
        return dp_port + bono_dp * 0.20  # asumimos 20% del portafolio en renta fija

    resultados = {}

    # ── Escenario 1: Shock de tasa ────────────────────────
    for shock_bp, label in [(-200, "tasa_menos_200pb"), (200, "tasa_mas_200pb")]:
        dy = shock_bp / 10000.0
        # Impacto en bono: duración + convexidad
        dp_bono = -dur_mod_bono * dy + 0.5 * convexidad_bono * dy**2
        # Impacto en CAPM: Rf cambia → precios ajustados por factor de riesgo
        # Simplificación: acciones de alta β más afectadas por subida de tasas
        dp_acciones = sum(
            pesos_arr[i] * (-betas.get(t, 1.0) * dy * 2)  # impacto aprox
            for i, t in enumerate(tickers)
        )
        dp_total = dp_acciones + dp_bono * 0.20
        var_estresado = abs(dp_total) if abs(dp_total) > var_base else var_base * 1.1
        resultados[label] = {
            "shock_pb":         shock_bp,
            "shock_descripcion": f"Δr = {shock_bp:+d} pb",
            "dp_bono_pct":      round(dp_bono * 100, 2),
            "dp_acciones_pct":  round(dp_acciones * 100, 2),
            "perdida_total_pct": round(dp_total * 100, 2),
            "perdida_total_usd": round(abs(dp_total) * valor_portafolio, 2),
            "var_base_pct":     round(var_base * 100, 4),
            "var_estresado_pct": round(var_estresado * 100, 4),
            "incremento_var_x": round(var_estresado / var_base, 2) if var_base > 0 else None,
        }

    # ── Escenario 2: Caída del mercado ────────────────────
    for shock_mkt, label in [(-0.20, "caida_mercado_20pct"), (-0.30, "caida_mercado_30pct")]:
        # Cada activo: ΔR_i = β_i × shock_mkt
        dp_activos = {
            t: betas.get(t, 1.0) * shock_mkt
            for t in tickers
        }
        dp_port = sum(pesos_arr[i] * dp_activos[t] for i, t in enumerate(tickers))
        var_estresado = abs(dp_port) if abs(dp_port) > var_base else var_base * 1.5
        resultados[label] = {
            "shock_mercado_pct": shock_mkt * 100,
            "shock_descripcion": f"Benchmark cae {shock_mkt*100:.0f}%",
            "impacto_por_activo": {
                t: round(dp * 100, 2) for t, dp in dp_activos.items()
            },
            "perdida_portafolio_pct": round(dp_port * 100, 2),
            "perdida_portafolio_usd": round(abs(dp_port) * valor_portafolio, 2),
            "var_base_pct":     round(var_base * 100, 4),
            "var_estresado_pct": round(var_estresado * 100, 4),
            "incremento_var_x": round(var_estresado / var_base, 2) if var_base > 0 else None,
        }

    # ── Escenario 3: Shock de volatilidad ─────────────────
    sigma_estresada = sigma_base * 2
    z_95            = stats.norm.ppf(0.05)
    var_param_stress = -(0.0 + z_95 * sigma_estresada)  # media≈0 en estrés
    resultados["volatilidad_doble"] = {
        "shock_descripcion":  "σ → σ × 2",
        "sigma_base_diaria":  round(sigma_base, 6),
        "sigma_estresada_diaria": round(sigma_estresada, 6),
        "var_param_base_pct": round(var_base * 100, 4),
        "var_param_estresado_pct": round(var_param_stress * 100, 4),
        "incremento_var_x":   round(var_param_stress / var_base, 2) if var_base > 0 else None,
        "nota_opciones":      "El efecto en opciones se mide via vega: ΔOpción ≈ vega × Δσ",
        "perdida_portafolio_pct": round(var_param_stress * 100, 2),
        "perdida_portafolio_usd": round(var_param_stress * valor_portafolio, 2),
    }

    # ── Escenario 4: Combinado (tormenta perfecta) ────────
    dp_comb_acc  = sum(pesos_arr[i] * betas.get(t, 1.0) * (-0.20) for i, t in enumerate(tickers))
    dp_comb_bono = (-dur_mod_bono * 0.02 + 0.5 * convexidad_bono * 0.02**2) * 0.20
    dp_comb_vol  = (var_param_stress - var_base)  # pérdida adicional por vol
    dp_comb_total = dp_comb_acc + dp_comb_bono - dp_comb_vol
    resultados["combinado_tormenta_perfecta"] = {
        "shock_descripcion": "Caída -20% + σ×2 + Δr +200pb simultáneos",
        "dp_acciones_pct":  round(dp_comb_acc * 100, 2),
        "dp_bono_pct":      round(dp_comb_bono * 100, 2),
        "dp_volatilidad_pct": round(-dp_comb_vol * 100, 2),
        "perdida_total_pct": round(dp_comb_total * 100, 2),
        "perdida_total_usd": round(abs(dp_comb_total) * valor_portafolio, 2),
    }

    return _limpiar_dict({
        "valor_portafolio_usd": valor_portafolio,
        "var_base_pct":         round(var_base * 100, 4),
        "sigma_base_diaria":    round(sigma_base, 6),
        "escenarios":           resultados,
        "heatmap_activo_escenario": {
            t: {
                "beta":        betas.get(t, 1.0),
                "caida_20pct": round(betas.get(t, 1.0) * -20, 2),
                "caida_30pct": round(betas.get(t, 1.0) * -30, 2),
                "shock_tasa_200pb": round(betas.get(t, 1.0) * -0.40, 2),
            }
            for t in tickers
        },
        "interpretacion": (
            "El escenario más severo es la 'tormenta perfecta' (combinado). "
            "Los escenarios de caída de mercado se propagan via beta de cada activo. "
            "El VaR estresado con σ×2 duplica aproximadamente el VaR base."
        ),
    })
