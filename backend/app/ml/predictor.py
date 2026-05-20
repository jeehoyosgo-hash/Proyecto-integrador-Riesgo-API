"""
predictor.py
============
Patrón Singleton para servir el modelo ML en FastAPI.

El modelo se carga UNA SOLA VEZ al primer request.
Llamadas sucesivas reutilizan la instancia: la línea de log
"[ModelPredictor] modelo cargado" solo aparece una vez en los logs.

Para verificar el Singleton en la sustentación:
  curl http://localhost:8000/predict -X POST -d '{"ticker":"AAPL","features":[...]}'  # × 3
  → El log de carga aparece solo en la primera llamada.
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Any, List
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "model.joblib"


class ModelPredictor:
    """
    Singleton: __new__ garantiza que solo existe UNA instancia del modelo.

    Implementación via variable de clase _instance.
    Compatible con múltiples workers de uvicorn si el modelo
    se carga dentro del proceso (no compartido entre procesos).
    """
    _instance = None
    _model: Any = None
    _model_version: str = "v1.0.0"
    _feature_names: List[str] = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._load_model()
        return cls._instance

    @classmethod
    def _load_model(cls):
        """Carga el modelo desde disco. Se ejecuta UNA sola vez."""
        if not MODEL_PATH.exists():
            logger.warning(
                f"[ModelPredictor] model.joblib no encontrado en {MODEL_PATH}. "
                f"Ejecuta: python -m app.ml.train"
            )
            cls._model = None
            return

        logger.info(f"[ModelPredictor] Cargando modelo desde {MODEL_PATH} ...")
        artifact = joblib.load(MODEL_PATH)

        # El artifact puede ser un Pipeline sklearn o un dict con metadata
        if isinstance(artifact, dict):
            cls._model          = artifact["model"]
            cls._feature_names  = artifact.get("feature_names", [])
            cls._model_version  = artifact.get("version", "v1.0.0")
        else:
            cls._model         = artifact
            cls._feature_names = []

        logger.info(
            f"[ModelPredictor] modelo cargado: {type(cls._model).__name__} | "
            f"versión: {cls._model_version}"
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Realiza la inferencia. El modelo ya está en memoria."""
        if self._model is None:
            raise RuntimeError(
                "Modelo no disponible. Ejecuta primero: python -m app.ml.train"
            )
        return self._model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Probabilidades de clase (si el modelo las soporta)."""
        if self._model is None:
            raise RuntimeError("Modelo no disponible.")
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(features)
        raise ValueError("El modelo no soporta predict_proba.")

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


# Función para Depends() en FastAPI
def get_predictor() -> ModelPredictor:
    """Retorna la instancia Singleton del predictor."""
    return ModelPredictor()
