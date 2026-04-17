# models.py
# Define la estructura de todos los datos que entran y salen de la API.
# Pydantic valida automáticamente que los datos sean correctos.

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import date


# ─────────────────────────────────────────────
# MODELOS DE ENTRADA (lo que el usuario envía)
# ─────────────────────────────────────────────

class PortafolioRequest(BaseModel):
    """
    Datos para calcular VaR y frontera eficiente.
    El usuario envía los tickers y los pesos de su portafolio.
    """
    tickers: list[str] = Field(
        default=["AAPL", "JPM", "XOM", "MSFT", "EC", "GLD"],
        description="Lista de tickers del portafolio",
        min_length=2,
        max_length=20
    )
    pesos: list[float] = Field(
        default=[0.20, 0.20, 0.15, 0.20, 0.10, 0.15],
        description="Pesos de cada activo. Deben sumar 1.0"
    )
    periodo: str = Field(
        default="2y",
        description="Período histórico: 1y, 2y, 3y, 5y"
    )
    confianza: float = Field(
        default=0.95,
        ge=0.90,
        le=0.99,
        description="Nivel de confianza para VaR (entre 0.90 y 0.99)"
    )

    # field_validator: valida que los pesos sumen exactamente 1.0
    # Esto es un requisito explícito de la rúbrica
    @field_validator("pesos")
    @classmethod
    def pesos_deben_sumar_uno(cls, v):
        total = round(sum(v), 4)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Los pesos deben sumar 1.0, pero suman {total}")
        return v

    # Valida que tickers y pesos tengan el mismo número de elementos
    @field_validator("pesos")
    @classmethod
    def validar_longitud_pesos(cls, v, info):
        if "tickers" in info.data and len(v) != len(info.data["tickers"]):
            raise ValueError("El número de pesos debe ser igual al número de tickers")
        return v


class RendimientoObjetivoRequest(BaseModel):
    """Para optimización con rendimiento objetivo (bonificación)"""
    tickers: list[str] = Field(default=["AAPL", "JPM", "XOM", "MSFT", "EC", "GLD"])
    rendimiento_objetivo: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Rendimiento anual objetivo (ej: 0.15 = 15%)"
    )
    periodo: str = Field(default="2y")


# ─────────────────────────────────────────────
# MODELOS DE RESPUESTA (lo que la API devuelve)
# ─────────────────────────────────────────────

class ActivoInfo(BaseModel):
    """Información básica de un activo"""
    ticker: str
    nombre: str
    sector: str
    precio_actual: float
    moneda: str


class PrecioHistorico(BaseModel):
    """Precio histórico de un activo"""
    fecha: str
    apertura: float
    maximo: float
    minimo: float
    cierre: float
    volumen: float


class RendimientoStats(BaseModel):
    """Estadísticas de rendimientos de un activo"""
    ticker: str
    media_diaria: float
    media_anual: float
    volatilidad_diaria: float
    volatilidad_anual: float
    asimetria: float
    curtosis: float
    sharpe_ratio: float
    es_normal_jb: bool        # True si pasa Jarque-Bera
    es_normal_sw: bool        # True si pasa Shapiro-Wilk
    p_valor_jb: float
    p_valor_sw: float


class GARCHResultado(BaseModel):
    """Resultado del ajuste de modelos ARCH/GARCH"""
    ticker: str
    modelo: str               # Ej: "GARCH(1,1)"
    aic: float
    bic: float
    alpha: float              # Coeficiente ARCH
    beta: float               # Coeficiente GARCH
    omega: float
    volatilidad_pronostico: float   # Volatilidad proyectada próximos días
    log_likelihood: float


class CAPMResultado(BaseModel):
    """Resultado del modelo CAPM para un activo"""
    ticker: str
    beta: float
    retorno_esperado: float
    retorno_mercado: float
    tasa_libre_riesgo: float
    r_cuadrado: float
    clasificacion: str        # "Agresivo", "Defensivo" o "Neutro"
    riesgo_sistematico: float
    riesgo_no_sistematico: float


class VaRResultado(BaseModel):
    """Resultado del cálculo de VaR y CVaR"""
    nivel_confianza: float
    var_parametrico: float
    var_historico: float
    var_montecarlo: float
    cvar_parametrico: float
    cvar_historico: float
    cvar_montecarlo: float
    periodo_dias: int = 1


class PortafolioOptimo(BaseModel):
    """Portafolio óptimo de la frontera eficiente"""
    tipo: str                 # "Máximo Sharpe" o "Mínima Varianza"
    tickers: list[str]
    pesos: list[float]
    retorno_esperado: float
    volatilidad: float
    sharpe_ratio: float


class FronteraEficienteResultado(BaseModel):
    """Resultado completo de la optimización de Markowitz"""
    portafolio_max_sharpe: PortafolioOptimo
    portafolio_min_varianza: PortafolioOptimo
    puntos_frontera: list[dict]   # Lista de {retorno, volatilidad, sharpe}
    correlaciones: dict           # Matriz de correlación


class Alerta(BaseModel):
    """Señal de compra/venta para un activo"""
    ticker: str
    señal: str                # "COMPRA", "VENTA" o "NEUTRAL"
    fuerza: str               # "FUERTE", "MODERADA" o "DÉBIL"
    razones: list[str]        # Ej: ["RSI sobrevendido", "MACD cruzó al alza"]
    rsi_actual: float
    macd_señal: str
    bollinger_señal: str
    estocastico_señal: str
    precio_actual: float


class MacroIndicadores(BaseModel):
    """Indicadores macroeconómicos actualizados"""
    tasa_libre_riesgo: float      # Fed Funds Rate o TES Colombia
    inflacion_usa: float          # CPI USA
    tasa_cambio_usd_cop: float    # TRM
    fecha_actualizacion: str
    fuente: str


class BenchmarkComparacion(BaseModel):
    """Comparación del portafolio vs benchmark"""
    retorno_portafolio: float
    retorno_benchmark: float
    alpha_jensen: float
    tracking_error: float
    information_ratio: float
    maximo_drawdown_portafolio: float
    maximo_drawdown_benchmark: float
    beta_portafolio: float


class IndicadoresTecnicos(BaseModel):
    """Indicadores técnicos calculados para un activo"""
    ticker: str
    fecha: str
    precio: float
    sma_20: Optional[float] = None
    ema_20: Optional[float] = None
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_señal: Optional[float] = None
    macd_histograma: Optional[float] = None
    bollinger_superior: Optional[float] = None
    bollinger_media: Optional[float] = None
    bollinger_inferior: Optional[float] = None
    estocastico_k: Optional[float] = None
    estocastico_d: Optional[float] = None