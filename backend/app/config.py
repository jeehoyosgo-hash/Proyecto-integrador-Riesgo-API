# config.py
# Este archivo maneja TODA la configuración del proyecto.
# BaseSettings lee automáticamente el archivo .env

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    Clase de configuración central.
    Cada variable aquí corresponde a una línea en el archivo .env
    """

    # API Keys externas
    fred_api_key: str = ""

    # Parámetros del portafolio (valores por defecto)
    tickers_default: list[str] = ["AAPL", "JPM", "XOM", "MSFT", "EC", "GLD"]
    benchmark_ticker: str = "SPY"        # S&P 500 como benchmark
    periodo_historico: str = "3y"        # 3 años de datos históricos

    # Parámetros de riesgo
    var_confianza_default: float = 0.95  # Nivel de confianza del VaR
    var_simulaciones: int = 10000        # Simulaciones Montecarlo
    riesgo_libre: float = 0.0525         # Tasa libre de riesgo por defecto (5.25%)

    # Parámetros técnicos
    sma_periodo: int = 20
    ema_periodo: int = 20
    rsi_periodo: int = 14
    bollinger_periodo: int = 20
    bollinger_std: float = 2.0

    class Config:
        env_file = ".env"           # Lee el archivo .env automáticamente
        env_file_encoding = "utf-8"


# lru_cache garantiza que solo se crea UNA instancia de Settings
# Esto es el patrón Singleton aplicado con decorador
@lru_cache()
def get_settings() -> Settings:
    """
    Retorna la instancia única de configuración.
    Se usa con Depends(get_settings) en los endpoints de FastAPI.
    """
    return Settings()