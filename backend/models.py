from pydantic import BaseModel, Field, field_validator
from typing import Optional, List


# ─────────────────────────────────────────────
# MODELOS DE ENTRADA (lo que el usuario ENVÍA)
# ─────────────────────────────────────────────

class PortafolioRequest(BaseModel):
    """
    Se usa en /var y /frontera-eficiente.
    El usuario envía una lista de acciones con sus pesos.
    Ejemplo: {"tickers": ["AAPL","MSFT"], "pesos": [0.6, 0.4]}
    """
    tickers: List[str] = Field(
        ...,
        min_length=2,
        description="Lista de símbolos. Ej: ['AAPL','MSFT','GOOGL']"
    )
    pesos: List[float] = Field(
        ...,
        description="Pesos del portafolio. DEBEN sumar 1.0. Ej: [0.5, 0.3, 0.2]"
    )
    fecha_inicio: Optional[str] = Field(
        default="2022-01-01",
        description="Fecha de inicio. Formato: YYYY-MM-DD"
    )
    fecha_fin: Optional[str] = Field(
        default=None,
        description="Fecha fin. Si no se envía, se usa la fecha de hoy."
    )
    nivel_confianza: Optional[float] = Field(
        default=0.95,
        ge=0.90,   # ge = mayor o igual que 0.90
        le=0.99,   # le = menor o igual que 0.99
        description="Nivel de confianza para VaR. Entre 0.90 y 0.99"
    )

    @field_validator('pesos')
    @classmethod
    def pesos_deben_sumar_uno(cls, pesos):
        total = sum(pesos)
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f'Los pesos deben sumar 1.0. Actualmente suman {total:.4f}'
            )
        return pesos

    @field_validator('tickers')
    @classmethod
    def tickers_a_mayusculas(cls, tickers):
        return [t.upper().strip() for t in tickers]


# ─────────────────────────────────────────────
# MODELOS DE SALIDA (lo que la API DEVUELVE)
# ─────────────────────────────────────────────

class HealthCheck(BaseModel):
    """Respuesta del endpoint raíz /"""
    status: str = "ok"
    mensaje: str
    version: str = "1.0.0"
    activos_disponibles: List[str]


class ErrorResponse(BaseModel):
    """Formato estándar para errores"""
    error: str
    detalle: str
    codigo: int
