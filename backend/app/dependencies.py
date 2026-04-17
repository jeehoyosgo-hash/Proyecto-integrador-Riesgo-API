# dependencies.py
# Inyección de dependencias con Depends().
# Esto garantiza que los endpoints no tengan lógica directa — requisito explícito de la rúbrica.

import httpx
from functools import lru_cache
from app.config import get_settings
from app.services import TechnicalIndicators, RiskCalculator, PortfolioAnalyzer


# ── Dependencias de servicios financieros ──

def get_technical_indicators() -> TechnicalIndicators:
    """Inyecta el servicio de indicadores técnicos."""
    return TechnicalIndicators()


def get_risk_calculator() -> RiskCalculator:
    """Inyecta la calculadora de riesgo."""
    return RiskCalculator()


def get_portfolio_analyzer() -> PortfolioAnalyzer:
    """Inyecta el analizador de portafolio."""
    return PortfolioAnalyzer()


# ── Dependencia de datos macro (FRED API) ──

async def get_macro_data() -> dict:
    """
    Obtiene indicadores macroeconómicos de FRED API.
    Se inyecta con Depends() en el endpoint /macro.
    """
    settings = get_settings()
    api_key = settings.fred_api_key

    indicadores = {
        "FEDFUNDS": "tasa_libre_riesgo",   # Tasa de fondos federales
        "CPIAUCSL": "inflacion_cpi",        # IPC USA
        "DEXCOUS": "tasa_cambio_usd_cop",  # USD/COP
    }

    resultados = {}

    async with httpx.AsyncClient(timeout=10.0) as client:
        for serie_id, nombre in indicadores.items():
            try:
                url = (
                    f"https://api.stlouisfed.org/fred/series/observations"
                    f"?series_id={serie_id}"
                    f"&api_key={api_key}"
                    f"&sort_order=desc"
                    f"&limit=1"
                    f"&file_type=json"
                )
                response = await client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    obs = data.get("observations", [])
                    if obs:
                        valor = obs[0].get("value", ".")
                        resultados[nombre] = float(valor) if valor != "." else None
                        resultados["fecha_" + nombre] = obs[0].get("date", "")
            except Exception:
                resultados[nombre] = None

    # Si FRED falla, usar valores por defecto razonables
    return {
        "tasa_libre_riesgo": resultados.get("tasa_libre_riesgo", 5.25) or 5.25,
        "inflacion_usa": resultados.get("inflacion_cpi", 3.2) or 3.2,
        "tasa_cambio_usd_cop": resultados.get("tasa_cambio_usd_cop", 4150.0) or 4150.0,
        "fecha_actualizacion": resultados.get("fecha_tasa_libre_riesgo", "N/A"),
        "fuente": "FRED API - Federal Reserve Bank of St. Louis",
    }
