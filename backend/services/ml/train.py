"""
train.py
========
Script de entrenamiento offline del modelo ML.

Propósito analítico elegido:
  Clasificación de régimen de mercado (alcista / bajista / lateral)
  usando features derivadas de los módulos de riesgo del proyecto.

Features de entrada (extraídas de los datos del proyecto):
  1. RSI (14 días)          — sobrecompra/sobreventa
  2. MACD histogram         — momentum
  3. Volatilidad EWMA       — régimen de riesgo
  4. Rendimiento 5 días     — momentum corto plazo
  5. Rendimiento 21 días    — momentum medio plazo
  6. %B Bollinger           — posición relativa del precio
  7. Estocástico %K         — momentum oscilador

Variable objetivo: rendimiento a 5 días:
  +1 = Alcista  (ret > +1%)
   0 = Lateral  (|ret| ≤ 1%)
  -1 = Bajista  (ret < -1%)

Ejecución:
  python -m app.ml.train
  python -m app.ml.train --ticker AAPL --periodo 3y
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "model.joblib"

FEATURE_NAMES = [
    "rsi_14",
    "macd_hist",
    "ewma_vol",
    "ret_5d",
    "ret_21d",
    "pct_b_bollinger",
    "estocastico_k",
]


def calcular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todas las features a partir de OHLCV."""
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # RSI (14)
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = delta.clip(upper=0).abs().ewm(com=13, adjust=False).mean()
    rsi   = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # MACD histogram
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist   = macd_line - signal_line

    # EWMA volatilidad (λ=0.94)
    log_ret  = np.log(close / close.shift(1))
    ewma_var = log_ret.pow(2).ewm(alpha=0.06, adjust=False).mean()
    ewma_vol = np.sqrt(ewma_var)

    # Rendimientos a 5 y 21 días
    ret_5d  = close.pct_change(5)
    ret_21d = close.pct_change(21)

    # Bollinger %B
    sma20   = close.rolling(20).mean()
    std20   = close.rolling(20).std()
    bb_sup  = sma20 + 2 * std20
    bb_inf  = sma20 - 2 * std20
    pct_b   = (close - bb_inf) / (bb_sup - bb_inf).replace(0, np.nan)

    # Estocástico %K
    low_14  = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    stoch_k = ((close - low_14) / (high_14 - low_14).replace(0, np.nan)) * 100

    features_df = pd.DataFrame({
        "rsi_14":            rsi,
        "macd_hist":         macd_hist,
        "ewma_vol":          ewma_vol,
        "ret_5d":            ret_5d,
        "ret_21d":           ret_21d,
        "pct_b_bollinger":   pct_b,
        "estocastico_k":     stoch_k,
    })

    return features_df


def calcular_target(close: pd.Series, horizonte: int = 5, umbral: float = 0.01) -> pd.Series:
    """
    Variable objetivo: régimen a horizonte días.
    +1 = Alcista, 0 = Lateral, -1 = Bajista
    """
    ret_futuro = close.shift(-horizonte) / close - 1
    target = ret_futuro.apply(
        lambda r: 1 if r > umbral else (-1 if r < -umbral else 0)
    )
    return target


def main(ticker: str = "AAPL", periodo: str = "3y"):
    logger.info(f"Descargando datos: {ticker} ({periodo})")
    df = yf.download(ticker, period=periodo, auto_adjust=True, progress=False)

    if df.empty:
        logger.error(f"No se encontraron datos para {ticker}")
        sys.exit(1)

    # Aplanar columnas MultiIndex si existen
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    logger.info(f"Datos descargados: {len(df)} filas")

    # Features y target
    features = calcular_features(df)
    target   = calcular_target(df["Close"])

    # Unir y limpiar NaN
    data = pd.concat([features, target.rename("y")], axis=1).dropna()
    logger.info(f"Dataset limpio: {len(data)} filas | clases: {data['y'].value_counts().to_dict()}")

    X = data[FEATURE_NAMES].values
    y = data["y"].values.astype(int)

    # Partición temporal (shuffle=False para respetar el orden)
    split = int(len(X) * 0.80)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # Pipeline: StandardScaler + RandomForest
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced",
        )),
    ])

    logger.info("Entrenando RandomForestClassifier (200 árboles)...")
    pipeline.fit(X_tr, y_tr)

    # Métricas
    y_pred = pipeline.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    logger.info(f"Accuracy (test set): {acc:.4f}")
    logger.info("\n" + classification_report(y_te, y_pred, target_names=["Bajista(-1)", "Lateral(0)", "Alcista(+1)"]))

    # Cross-validation temporal (TimeSeriesSplit)
    from sklearn.model_selection import TimeSeriesSplit
    tscv   = TimeSeriesSplit(n_splits=5)
    cv_acc = cross_val_score(pipeline, X, y, cv=tscv, scoring="accuracy")
    logger.info(f"CV accuracy (TimeSeriesSplit): {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

    # Guardar el artefacto
    artifact = {
        "model":         pipeline,
        "feature_names": FEATURE_NAMES,
        "version":       "v1.0.0",
        "ticker_train":  ticker,
        "periodo_train": periodo,
        "n_train":       len(X_tr),
        "n_test":        len(X_te),
        "accuracy_test": round(float(acc), 4),
        "cv_accuracy":   round(float(cv_acc.mean()), 4),
    }

    joblib.dump(artifact, MODEL_PATH)
    logger.info(f"Modelo guardado en: {MODEL_PATH}")
    logger.info(
        f"Para servir: uvicorn app.main:app --reload\n"
        f"Llamar a: POST /predict con features={FEATURE_NAMES}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",  default="AAPL", help="Ticker para entrenamiento")
    parser.add_argument("--periodo", default="3y",   help="Período histórico: 1y, 2y, 3y, 5y")
    args = parser.parse_args()
    main(args.ticker, args.periodo)
