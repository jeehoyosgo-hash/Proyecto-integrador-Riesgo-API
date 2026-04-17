# services.py
# Motor de cálculo financiero del proyecto.
# Contiene las clases: TechnicalIndicators, RiskCalculator, PortfolioAnalyzer

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import yfinance as yf
from arch import arch_model
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# CLASE 1: INDICADORES TÉCNICOS
# ─────────────────────────────────────────────────────────

class TechnicalIndicators:
    """Calcula todos los indicadores técnicos para un activo."""

    @staticmethod
    def obtener_precios(ticker: str, periodo: str = "2y") -> pd.DataFrame:
        """Descarga precios históricos desde Yahoo Finance."""
        try:
            data = yf.download(ticker, period=periodo, auto_adjust=True, progress=False)
            if data.empty:
                raise ValueError(f"No se encontraron datos para {ticker}")
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
            return data
        except Exception as e:
            raise ValueError(f"Error descargando {ticker}: {str(e)}")

    @staticmethod
    def sma(precios: pd.Series, periodo: int = 20) -> pd.Series:
        """Media Móvil Simple."""
        return precios.rolling(window=periodo).mean()

    @staticmethod
    def ema(precios: pd.Series, periodo: int = 20) -> pd.Series:
        """Media Móvil Exponencial."""
        return precios.ewm(span=periodo, adjust=False).mean()

    @staticmethod
    def rsi(precios: pd.Series, periodo: int = 14) -> pd.Series:
        """Índice de Fuerza Relativa (RSI)."""
        delta = precios.diff()
        ganancia = delta.where(delta > 0, 0).rolling(window=periodo).mean()
        perdida = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        rs = ganancia / perdida
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(precios: pd.Series, rapida: int = 12, lenta: int = 26, señal: int = 9):
        """MACD, Señal e Histograma."""
        ema_rapida = precios.ewm(span=rapida, adjust=False).mean()
        ema_lenta = precios.ewm(span=lenta, adjust=False).mean()
        macd_line = ema_rapida - ema_lenta
        signal_line = macd_line.ewm(span=señal, adjust=False).mean()
        histograma = macd_line - signal_line
        return macd_line, signal_line, histograma

    @staticmethod
    def bollinger(precios: pd.Series, periodo: int = 20, std: float = 2.0):
        """Bandas de Bollinger."""
        media = precios.rolling(window=periodo).mean()
        desviacion = precios.rolling(window=periodo).std()
        superior = media + (std * desviacion)
        inferior = media - (std * desviacion)
        return superior, media, inferior

    @staticmethod
    def estocastico(df: pd.DataFrame, periodo: int = 14):
        """Oscilador Estocástico %K y %D."""
        low_min = df["Low"].rolling(window=periodo).min()
        high_max = df["High"].rolling(window=periodo).max()
        k = 100 * (df["Close"] - low_min) / (high_max - low_min)
        d = k.rolling(window=3).mean()
        return k, d

    def calcular_todos(self, ticker: str, periodo: str = "2y") -> dict:
        """Calcula todos los indicadores técnicos de un activo."""
        df = self.obtener_precios(ticker, periodo)
        precios = df["Close"]

        sma = self.sma(precios)
        ema = self.ema(precios)
        rsi = self.rsi(precios)
        macd_l, macd_s, macd_h = self.macd(precios)
        bb_sup, bb_med, bb_inf = self.bollinger(precios)
        esto_k, esto_d = self.estocastico(df)

        resultado = []
        for fecha in df.index[-252:]:  # Último año
            idx = df.index.get_loc(fecha)
            resultado.append({
                "fecha": str(fecha.date()),
                "precio": round(float(precios.iloc[idx]), 2),
                "sma_20": round(float(sma.iloc[idx]), 2) if not pd.isna(sma.iloc[idx]) else None,
                "ema_20": round(float(ema.iloc[idx]), 2) if not pd.isna(ema.iloc[idx]) else None,
                "rsi_14": round(float(rsi.iloc[idx]), 2) if not pd.isna(rsi.iloc[idx]) else None,
                "macd": round(float(macd_l.iloc[idx]), 4) if not pd.isna(macd_l.iloc[idx]) else None,
                "macd_señal": round(float(macd_s.iloc[idx]), 4) if not pd.isna(macd_s.iloc[idx]) else None,
                "macd_histograma": round(float(macd_h.iloc[idx]), 4) if not pd.isna(macd_h.iloc[idx]) else None,
                "bollinger_superior": round(float(bb_sup.iloc[idx]), 2) if not pd.isna(bb_sup.iloc[idx]) else None,
                "bollinger_media": round(float(bb_med.iloc[idx]), 2) if not pd.isna(bb_med.iloc[idx]) else None,
                "bollinger_inferior": round(float(bb_inf.iloc[idx]), 2) if not pd.isna(bb_inf.iloc[idx]) else None,
                "estocastico_k": round(float(esto_k.iloc[idx]), 2) if not pd.isna(esto_k.iloc[idx]) else None,
                "estocastico_d": round(float(esto_d.iloc[idx]), 2) if not pd.isna(esto_d.iloc[idx]) else None,
            })
        return resultado


# ─────────────────────────────────────────────────────────
# CLASE 2: CALCULADORA DE RIESGO
# ─────────────────────────────────────────────────────────

class RiskCalculator:
    """Calcula VaR, CVaR, CAPM y modelos ARCH/GARCH."""

    def rendimientos(self, ticker: str, periodo: str = "2y") -> pd.Series:
        """Retorna rendimientos logarítmicos diarios."""
        df = TechnicalIndicators.obtener_precios(ticker, periodo)
        return np.log(df["Close"] / df["Close"].shift(1)).dropna()

    def estadisticas_rendimientos(self, ticker: str, periodo: str = "2y") -> dict:
        """Estadísticas descriptivas completas de los rendimientos."""
        ret = self.rendimientos(ticker, periodo)
        ret_simple = TechnicalIndicators.obtener_precios(ticker, periodo)["Close"].pct_change().dropna()

        # Pruebas de normalidad
        jb_stat, jb_pval = stats.jarque_bera(ret)
        sw_stat, sw_pval = stats.shapiro(ret[-200:])  # Shapiro max 5000 obs

        # Sharpe ratio anualizado (asumiendo rf=5.25%)
        rf_diaria = 0.0525 / 252
        sharpe = (ret.mean() - rf_diaria) / ret.std() * np.sqrt(252)

        return {
            "ticker": ticker,
            "media_diaria": round(float(ret.mean()), 6),
            "media_anual": round(float(ret.mean() * 252), 4),
            "volatilidad_diaria": round(float(ret.std()), 6),
            "volatilidad_anual": round(float(ret.std() * np.sqrt(252)), 4),
            "asimetria": round(float(stats.skew(ret)), 4),
            "curtosis": round(float(stats.kurtosis(ret)), 4),
            "sharpe_ratio": round(float(sharpe), 4),
            "es_normal_jb": bool(jb_pval > 0.05),
            "es_normal_sw": bool(sw_pval > 0.05),
            "p_valor_jb": round(float(jb_pval), 6),
            "p_valor_sw": round(float(sw_pval), 6),
            "rendimientos_simples": ret_simple.tolist(),
            "rendimientos_log": ret.tolist(),
            "fechas": [str(d.date()) for d in ret.index],
        }

    def garch_analysis(self, ticker: str, periodo: str = "2y") -> list:
        """Ajusta 3 modelos ARCH/GARCH y los compara."""
        ret = self.rendimientos(ticker, periodo) * 100  # Escalar para GARCH

        modelos_config = [
            ("ARCH(1)", arch_model(ret, vol="ARCH", p=1)),
            ("GARCH(1,1)", arch_model(ret, vol="Garch", p=1, q=1)),
            ("EGARCH(1,1)", arch_model(ret, vol="EGARCH", p=1, q=1)),
        ]

        resultados = []
        for nombre, modelo in modelos_config:
            try:
                fit = modelo.fit(disp="off")
                forecast = fit.forecast(horizon=5)
                vol_pronostico = float(np.sqrt(forecast.variance.values[-1, -1])) / 100

                params = fit.params
                resultado = {
                    "ticker": ticker,
                    "modelo": nombre,
                    "aic": round(float(fit.aic), 4),
                    "bic": round(float(fit.bic), 4),
                    "log_likelihood": round(float(fit.loglikelihood), 4),
                    "volatilidad_pronostico": round(vol_pronostico, 6),
                    "omega": round(float(params.get("omega", 0)), 6),
                    "alpha": round(float(params.get("alpha[1]", params.get("alpha", 0))), 4),
                    "beta": round(float(params.get("beta[1]", params.get("beta", 0))), 4),
                    "residuos_estandarizados": fit.std_resid.tolist(),
                }
                resultados.append(resultado)
            except Exception as e:
                resultados.append({"ticker": ticker, "modelo": nombre, "error": str(e)})

        return resultados

    def capm(self, ticker: str, benchmark: str = "SPY", periodo: str = "2y", rf: float = 0.0525) -> dict:
        """Calcula Beta y retorno esperado CAPM."""
        ret_activo = self.rendimientos(ticker, periodo)
        ret_mercado = self.rendimientos(benchmark, periodo)

        # Alinear fechas
        df = pd.DataFrame({"activo": ret_activo, "mercado": ret_mercado}).dropna()

        # Regresión lineal para obtener Beta
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df["mercado"], df["activo"]
        )

        beta = slope
        rf_diaria = rf / 252
        retorno_mercado_anual = float(df["mercado"].mean() * 252)
        retorno_esperado = rf + beta * (retorno_mercado_anual - rf)

        # Clasificación
        if beta > 1.1:
            clasificacion = "Agresivo"
        elif beta < 0.9:
            clasificacion = "Defensivo"
        else:
            clasificacion = "Neutro"

        # Riesgo sistemático vs no sistemático
        var_mercado = float(df["mercado"].var() * 252)
        var_activo = float(df["activo"].var() * 252)
        riesgo_sistematico = beta**2 * var_mercado
        riesgo_no_sistematico = var_activo - riesgo_sistematico

        return {
            "ticker": ticker,
            "beta": round(beta, 4),
            "retorno_esperado": round(retorno_esperado, 4),
            "retorno_mercado": round(retorno_mercado_anual, 4),
            "tasa_libre_riesgo": rf,
            "r_cuadrado": round(r_value**2, 4),
            "clasificacion": clasificacion,
            "riesgo_sistematico": round(riesgo_sistematico, 6),
            "riesgo_no_sistematico": round(max(riesgo_no_sistematico, 0), 6),
            "datos_regresion": {
                "x": df["mercado"].tolist(),
                "y": df["activo"].tolist(),
            }
        }

    def var_cvar(self, tickers: list, pesos: list, confianza: float = 0.95,
                 periodo: str = "2y", simulaciones: int = 10000) -> dict:
        """Calcula VaR y CVaR con 3 métodos."""
        # Descargar rendimientos de todos los activos
        rets = {}
        for t in tickers:
            try:
                rets[t] = self.rendimientos(t, periodo)
            except:
                pass

        df_rets = pd.DataFrame(rets).dropna()
        pesos_arr = np.array(pesos[:len(df_rets.columns)])
        pesos_arr = pesos_arr / pesos_arr.sum()

        # Rendimientos del portafolio
        ret_portafolio = df_rets.values @ pesos_arr

        # ── Método 1: Paramétrico (Normal) ──
        mu = np.mean(ret_portafolio)
        sigma = np.std(ret_portafolio)
        z = stats.norm.ppf(1 - confianza)
        var_param = -(mu + z * sigma)
        cvar_param = -(mu - sigma * stats.norm.pdf(z) / (1 - confianza))

        # ── Método 2: Simulación Histórica ──
        var_hist = float(-np.percentile(ret_portafolio, (1 - confianza) * 100))
        losses_hist = ret_portafolio[ret_portafolio < -var_hist]
        cvar_hist = float(-losses_hist.mean()) if len(losses_hist) > 0 else var_hist

        # ── Método 3: Montecarlo ──
        media_vec = df_rets.mean().values
        cov_matrix = df_rets.cov().values
        sim_rets = np.random.multivariate_normal(media_vec, cov_matrix, simulaciones)
        sim_port = sim_rets @ pesos_arr
        var_mc = float(-np.percentile(sim_port, (1 - confianza) * 100))
        losses_mc = sim_port[sim_port < -var_mc]
        cvar_mc = float(-losses_mc.mean()) if len(losses_mc) > 0 else var_mc

        return {
            "nivel_confianza": confianza,
            "var_parametrico": round(var_param, 6),
            "var_historico": round(var_hist, 6),
            "var_montecarlo": round(var_mc, 6),
            "cvar_parametrico": round(cvar_param, 6),
            "cvar_historico": round(cvar_hist, 6),
            "cvar_montecarlo": round(cvar_mc, 6),
            "periodo_dias": 1,
            "simulaciones_mc": simulaciones,
            "rendimientos_portafolio": ret_portafolio.tolist(),
            "simulaciones_portafolio": sim_port.tolist(),
        }


# ─────────────────────────────────────────────────────────
# CLASE 3: ANALIZADOR DE PORTAFOLIO (MARKOWITZ)
# ─────────────────────────────────────────────────────────

class PortfolioAnalyzer:
    """Optimización de portafolio con la teoría de Markowitz."""

    def __init__(self):
        self.risk_calc = RiskCalculator()

    def _obtener_matriz_rendimientos(self, tickers: list, periodo: str = "2y") -> pd.DataFrame:
        """Descarga y alinea rendimientos de todos los activos."""
        rets = {}
        for t in tickers:
            try:
                rets[t] = self.risk_calc.rendimientos(t, periodo)
            except:
                pass
        return pd.DataFrame(rets).dropna()

    def frontera_eficiente(self, tickers: list, periodo: str = "2y",
                           n_portafolios: int = 10000, rf: float = 0.0525) -> dict:
        """Genera la frontera eficiente con 10,000 portafolios simulados."""
        df = self._obtener_matriz_rendimientos(tickers, periodo)
        media = df.mean() * 252
        cov = df.cov() * 252
        n = len(tickers)

        # Simular portafolios aleatorios
        resultados = []
        for _ in range(n_portafolios):
            w = np.random.random(n)
            w = w / w.sum()
            ret = float(w @ media.values)
            vol = float(np.sqrt(w @ cov.values @ w))
            sharpe = (ret - rf) / vol if vol > 0 else 0
            resultados.append({
                "retorno": round(ret, 4),
                "volatilidad": round(vol, 4),
                "sharpe": round(sharpe, 4),
                "pesos": w.tolist()
            })

        resultados_df = pd.DataFrame(resultados)

        # Portafolio máximo Sharpe
        idx_max_sharpe = resultados_df["sharpe"].idxmax()
        max_sharpe = resultados_df.iloc[idx_max_sharpe]

        # Portafolio mínima varianza
        idx_min_var = resultados_df["volatilidad"].idxmin()
        min_var = resultados_df.iloc[idx_min_var]

        # Correlaciones
        correlaciones = df.corr().round(4).to_dict()

        return {
            "portafolio_max_sharpe": {
                "tipo": "Máximo Sharpe",
                "tickers": tickers,
                "pesos": [round(p, 4) for p in max_sharpe["pesos"]],
                "retorno_esperado": round(float(max_sharpe["retorno"]), 4),
                "volatilidad": round(float(max_sharpe["volatilidad"]), 4),
                "sharpe_ratio": round(float(max_sharpe["sharpe"]), 4),
            },
            "portafolio_min_varianza": {
                "tipo": "Mínima Varianza",
                "tickers": tickers,
                "pesos": [round(p, 4) for p in min_var["pesos"]],
                "retorno_esperado": round(float(min_var["retorno"]), 4),
                "volatilidad": round(float(min_var["volatilidad"]), 4),
                "sharpe_ratio": round(float(min_var["sharpe"]), 4),
            },
            "puntos_frontera": resultados[:500],  # 500 puntos para el gráfico
            "correlaciones": correlaciones,
        }

    def señales_alertas(self, tickers: list) -> list:
        """Genera señales de compra/venta basadas en indicadores técnicos."""
        ti = TechnicalIndicators()
        alertas = []

        for ticker in tickers:
            try:
                df = ti.obtener_precios(ticker, "6mo")
                precios = df["Close"]

                rsi = ti.rsi(precios).iloc[-1]
                macd_l, macd_s, macd_h = ti.macd(precios)
                bb_sup, bb_med, bb_inf = ti.bollinger(precios)
                esto_k, esto_d = ti.estocastico(df)

                precio_actual = float(precios.iloc[-1])
                macd_actual = float(macd_l.iloc[-1])
                macd_señal_val = float(macd_s.iloc[-1])
                macd_prev = float(macd_l.iloc[-2])
                macd_señal_prev = float(macd_s.iloc[-2])

                razones = []
                puntos_compra = 0
                puntos_venta = 0

                # RSI
                rsi_val = float(rsi) if not pd.isna(rsi) else 50
                if rsi_val < 30:
                    razones.append(f"RSI sobrevendido ({rsi_val:.1f})")
                    puntos_compra += 2
                elif rsi_val > 70:
                    razones.append(f"RSI sobrecomprado ({rsi_val:.1f})")
                    puntos_venta += 2

                # MACD cruce
                if macd_prev < macd_señal_prev and macd_actual > macd_señal_val:
                    razones.append("MACD cruzó al alza (señal de compra)")
                    puntos_compra += 2
                    macd_señal_str = "ALCISTA"
                elif macd_prev > macd_señal_prev and macd_actual < macd_señal_val:
                    razones.append("MACD cruzó a la baja (señal de venta)")
                    puntos_venta += 2
                    macd_señal_str = "BAJISTA"
                else:
                    macd_señal_str = "NEUTRAL"

                # Bollinger
                bb_sup_val = float(bb_sup.iloc[-1])
                bb_inf_val = float(bb_inf.iloc[-1])
                if precio_actual < bb_inf_val:
                    razones.append("Precio bajo la banda inferior de Bollinger")
                    puntos_compra += 1
                    bollinger_señal_str = "SOBREVENDIDO"
                elif precio_actual > bb_sup_val:
                    razones.append("Precio sobre la banda superior de Bollinger")
                    puntos_venta += 1
                    bollinger_señal_str = "SOBRECOMPRADO"
                else:
                    bollinger_señal_str = "NEUTRAL"

                # Estocástico
                k_val = float(esto_k.iloc[-1]) if not pd.isna(esto_k.iloc[-1]) else 50
                d_val = float(esto_d.iloc[-1]) if not pd.isna(esto_d.iloc[-1]) else 50
                if k_val < 20 and d_val < 20:
                    razones.append(f"Estocástico en zona de sobreventa (K={k_val:.1f})")
                    puntos_compra += 1
                    esto_señal = "SOBREVENTA"
                elif k_val > 80 and d_val > 80:
                    razones.append(f"Estocástico en zona de sobrecompra (K={k_val:.1f})")
                    puntos_venta += 1
                    esto_señal = "SOBRECOMPRA"
                else:
                    esto_señal = "NEUTRAL"

                # Señal final
                if puntos_compra > puntos_venta:
                    señal = "COMPRA"
                    fuerza = "FUERTE" if puntos_compra >= 4 else "MODERADA"
                elif puntos_venta > puntos_compra:
                    señal = "VENTA"
                    fuerza = "FUERTE" if puntos_venta >= 4 else "MODERADA"
                else:
                    señal = "NEUTRAL"
                    fuerza = "DÉBIL"

                if not razones:
                    razones.append("Sin señales técnicas claras en este momento")

                alertas.append({
                    "ticker": ticker,
                    "señal": señal,
                    "fuerza": fuerza,
                    "razones": razones,
                    "rsi_actual": round(rsi_val, 2),
                    "macd_señal": macd_señal_str,
                    "bollinger_señal": bollinger_señal_str,
                    "estocastico_señal": esto_señal,
                    "precio_actual": round(precio_actual, 2),
                })

            except Exception as e:
                alertas.append({
                    "ticker": ticker,
                    "señal": "ERROR",
                    "fuerza": "N/A",
                    "razones": [str(e)],
                    "rsi_actual": 0,
                    "macd_señal": "N/A",
                    "bollinger_señal": "N/A",
                    "estocastico_señal": "N/A",
                    "precio_actual": 0,
                })

        return alertas

    def benchmark_comparacion(self, tickers: list, pesos: list,
                               benchmark: str = "SPY", periodo: str = "2y",
                               rf: float = 0.0525) -> dict:
        """Compara el portafolio óptimo contra el benchmark."""
        rc = RiskCalculator()

        # Rendimientos del portafolio
        rets = {}
        for t in tickers:
            try:
                rets[t] = rc.rendimientos(t, periodo)
            except:
                pass
        df = pd.DataFrame(rets).dropna()
        pesos_arr = np.array(pesos[:len(df.columns)]) / sum(pesos[:len(df.columns)])
        ret_port = df.values @ pesos_arr

        # Rendimientos del benchmark
        ret_bench = rc.rendimientos(benchmark, periodo)
        df_bench = pd.DataFrame({"port": ret_port, "bench": ret_bench},
                                 index=df.index).dropna()

        # Métricas
        ret_port_anual = float(df_bench["port"].mean() * 252)
        ret_bench_anual = float(df_bench["bench"].mean() * 252)

        # Beta del portafolio
        slope, _, r, _, _ = stats.linregress(df_bench["bench"], df_bench["port"])
        beta_port = slope

        # Alpha de Jensen
        alpha_jensen = ret_port_anual - (rf + beta_port * (ret_bench_anual - rf))

        # Tracking Error e Information Ratio
        diff = df_bench["port"] - df_bench["bench"]
        tracking_error = float(diff.std() * np.sqrt(252))
        information_ratio = float(diff.mean() * 252 / tracking_error) if tracking_error > 0 else 0

        # Máximo Drawdown
        def max_drawdown(rets_series):
            cum = (1 + rets_series).cumprod()
            rolling_max = cum.cummax()
            drawdown = (cum - rolling_max) / rolling_max
            return float(drawdown.min())

        mdd_port = max_drawdown(df_bench["port"])
        mdd_bench = max_drawdown(df_bench["bench"])

        # Retorno acumulado base 100
        acum_port = ((1 + df_bench["port"]).cumprod() * 100).tolist()
        acum_bench = ((1 + df_bench["bench"]).cumprod() * 100).tolist()
        fechas = [str(d.date()) for d in df_bench.index]

        return {
            "retorno_portafolio": round(ret_port_anual, 4),
            "retorno_benchmark": round(ret_bench_anual, 4),
            "alpha_jensen": round(alpha_jensen, 4),
            "tracking_error": round(tracking_error, 4),
            "information_ratio": round(information_ratio, 4),
            "maximo_drawdown_portafolio": round(mdd_port, 4),
            "maximo_drawdown_benchmark": round(mdd_bench, 4),
            "beta_portafolio": round(beta_port, 4),
            "acumulado_portafolio": acum_port,
            "acumulado_benchmark": acum_bench,
            "fechas": fechas,
        }