# dependencies.py
import httpx
from app.config import get_settings


def get_technical_indicators():
    return None


def get_risk_calculator():
    return None


def get_portfolio_analyzer():
    return None


async def get_macro_data() -> dict:
    settings = get_settings()
    api_key = settings.fred_api_key

    indicadores = {
        "FEDFUNDS": "tasa_libre_riesgo",
        "CPIAUCSL": "inflacion_cpi",
    }
    resultados = {}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            for serie_id, nombre in indicadores.items():
                try:
                    url = (
                        "https://api.stlouisfed.org/fred/series/observations"
                        f"?series_id={serie_id}&api_key={api_key}"
                        "&sort_order=desc&limit=1&file_type=json"
                    )
                    response = await client.get(url)
                    if response.status_code == 200:
                        obs = response.json().get("observations", [])
                        if obs:
                            valor = obs[0].get("value", ".")
                            resultados[nombre] = float(valor) if valor != "." else None
                except Exception:
                    resultados[nombre] = None
    except Exception:
        pass

    return {
        "tasa_libre_riesgo":   resultados.get("tasa_libre_riesgo") or 5.25,
        "inflacion_usa":       resultados.get("inflacion_cpi") or 3.2,
        "tasa_cambio_usd_cop": 4150.0,
        "fecha_actualizacion": "N/A",
        "fuente":              "FRED API",
    }