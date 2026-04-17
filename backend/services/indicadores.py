import pandas as pd
import numpy as np
from typing import Optional
from services.datos import descargar_precios


def limpiar_valor(v):
    """
    Convierte NaN e Infinity a None para que JSON no falle.
    NaN y Infinity aparecen en los primeros días cuando no hay
    suficiente historia para calcular el indicador.
    """
    if v is None:
        return None
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None


def limpiar_dataframe(df: pd.DataFrame) -> list:
    """
    Convierte un DataFrame a lista de diccionarios limpiando
    todos los NaN e Infinity de cada fila.
    """
    registros = []
    for _, fila in df.iterrows():
        registro = {}
        for col, val in fila.items():
            if col == "fecha":
                registro[col] = str(val)
            else:
                registro[col] = limpiar_valor(val)
        registros.append(registro)
    return registros


def calcular_sma(precios: pd.Series, ventana: int) -> pd.Series:
    """
    SMA — Media Móvil Simple.
    Promedio de los últimos N días de precio de cierre.
    Señal COMPRA cuando precio cruza SMA hacia arriba.
    Señal VENTA cuando precio cruza SMA hacia abajo.
    """
    return precios.rolling(window=ventana).mean().round(4)


def calcular_ema(precios: pd.Series, ventana: int) -> pd.Series:
    """
    EMA — Media Móvil Exponencial.
    Da más peso a los días recientes que la SMA.
    Reacciona más rápido a cambios de precio.
    """
    return precios.ewm(span=ventana, adjust=False).mean().round(4)


def calcular_rsi(precios: pd.Series, ventana: int = 14) -> pd.Series:
    """
    RSI — Índice de Fuerza Relativa. Va de 0 a 100.
    RSI > 70 → sobrecomprada (posible VENTA)
    RSI < 30 → sobrevendida  (posible COMPRA)
    """
    delta    = precios.diff()
    ganancias = delta.clip(lower=0)
    perdidas  = delta.clip(upper=0).abs()

    media_ganancias = ganancias.ewm(com=ventana - 1, adjust=False).mean()
    media_perdidas  = perdidas.ewm(com=ventana - 1, adjust=False).mean()

    rs  = media_ganancias / media_perdidas
    rsi = 100 - (100 / (1 + rs))
    return rsi.round(2)


def calcular_macd(
    precios: pd.Series,
    rapida: int = 12,
    lenta: int = 26,
    señal: int = 9,
) -> pd.DataFrame:
    """
    MACD — Detecta cambios en la tendencia.
    MACD > Signal → tendencia alcista (COMPRA)
    MACD < Signal → tendencia bajista (VENTA)
    """
    ema_rapida  = calcular_ema(precios, rapida)
    ema_lenta   = calcular_ema(precios, lenta)
    macd_line   = (ema_rapida - ema_lenta).round(4)
    signal_line = macd_line.ewm(span=señal, adjust=False).mean().round(4)
    histograma  = (macd_line - signal_line).round(4)

    return pd.DataFrame({
        "macd":       macd_line,
        "macd_señal": signal_line,
        "macd_hist":  histograma,
    })


def calcular_bollinger(
    precios: pd.Series,
    ventana: int = 20,
    num_desviaciones: float = 2.0,
) -> pd.DataFrame:
    """
    Bandas de Bollinger — miden la volatilidad.
    Precio toca banda superior → posible VENTA
    Precio toca banda inferior → posible COMPRA
    """
    sma             = calcular_sma(precios, ventana)
    std             = precios.rolling(window=ventana).std()
    banda_superior  = (sma + num_desviaciones * std).round(4)
    banda_inferior  = (sma - num_desviaciones * std).round(4)
    ancho           = banda_superior - banda_inferior
    pct_b           = ((precios - banda_inferior) / ancho).round(4)

    return pd.DataFrame({
        "boll_superior": banda_superior,
        "boll_media":    sma,
        "boll_inferior": banda_inferior,
        "boll_pct_b":    pct_b,
    })


def calcular_estocastico(
    df: pd.DataFrame,
    ventana_k: int = 14,
    ventana_d: int = 3,
) -> pd.DataFrame:
    """
    Oscilador Estocástico. Va de 0 a 100.
    %K > 80 → sobrecomprado (VENTA)
    %K < 20 → sobrevendido  (COMPRA)
    """
    minimo_n = df["minimo"].rolling(window=ventana_k).min()
    maximo_n = df["maximo"].rolling(window=ventana_k).max()
    k = ((df["cierre"] - minimo_n) / (maximo_n - minimo_n) * 100).round(2)
    d = k.rolling(window=ventana_d).mean().round(2)

    return pd.DataFrame({"esto_k": k, "esto_d": d})


def generar_señales(ultimo: dict) -> list:
    """
    Genera señales de compra/venta del último día disponible.
    """
    señales = []

    # Señal RSI
    rsi = ultimo.get("rsi_14")
    if rsi is not None:
        if rsi < 30:
            señales.append({
                "tipo":        "COMPRA",
                "indicador":   "RSI",
                "valor":       rsi,
                "descripcion": f"RSI={rsi:.1f} — acción sobrevendida (< 30)",
            })
        elif rsi > 70:
            señales.append({
                "tipo":        "VENTA",
                "indicador":   "RSI",
                "valor":       rsi,
                "descripcion": f"RSI={rsi:.1f} — acción sobrecomprada (> 70)",
            })

    # Señal MACD
    macd  = ultimo.get("macd")
    señal = ultimo.get("macd_señal")
    if macd is not None and señal is not None:
        if macd > señal:
            señales.append({
                "tipo":        "COMPRA",
                "indicador":   "MACD",
                "valor":       round(macd - señal, 4),
                "descripcion": "MACD sobre línea de señal — tendencia alcista",
            })
        else:
            señales.append({
                "tipo":        "VENTA",
                "indicador":   "MACD",
                "valor":       round(macd - señal, 4),
                "descripcion": "MACD bajo línea de señal — tendencia bajista",
            })

    # Señal Bollinger
    pct_b = ultimo.get("boll_pct_b")
    if pct_b is not None:
        if pct_b > 0.8:
            señales.append({
                "tipo":        "VENTA",
                "indicador":   "Bollinger",
                "valor":       pct_b,
                "descripcion": f"%B={pct_b:.2f} — precio cerca de banda superior",
            })
        elif pct_b < 0.2:
            señales.append({
                "tipo":        "COMPRA",
                "indicador":   "Bollinger",
                "valor":       pct_b,
                "descripcion": f"%B={pct_b:.2f} — precio cerca de banda inferior",
            })

    return señales


def calcular_todos_indicadores(
    ticker: str,
    fecha_inicio: str = "2022-01-01",
    fecha_fin: Optional[str] = None,
) -> dict:
    """
    Función principal: descarga precios y calcula TODOS los indicadores.
    """
    # 1. Descargar precios
    df = descargar_precios(ticker, fecha_inicio, fecha_fin)
    precios = df["cierre"]

    # 2. Calcular indicadores
    sma_20   = calcular_sma(precios, 20)
    sma_50   = calcular_sma(precios, 50)
    sma_200  = calcular_sma(precios, 200)
    ema_20   = calcular_ema(precios, 20)
    ema_50   = calcular_ema(precios, 50)
    rsi_14   = calcular_rsi(precios, 14)
    macd_df  = calcular_macd(precios)
    boll_df  = calcular_bollinger(precios)
    esto_df  = calcular_estocastico(df)

    # 3. Combinar en un DataFrame
    resultado = pd.DataFrame({
        "fecha":        df["fecha"],
        "cierre":       precios.round(2),
        "sma_20":       sma_20,
        "sma_50":       sma_50,
        "sma_200":      sma_200,
        "ema_20":       ema_20,
        "ema_50":       ema_50,
        "rsi_14":       rsi_14,
        "macd":         macd_df["macd"],
        "macd_señal":   macd_df["macd_señal"],
        "macd_hist":    macd_df["macd_hist"],
        "boll_superior":boll_df["boll_superior"],
        "boll_media":   boll_df["boll_media"],
        "boll_inferior":boll_df["boll_inferior"],
        "boll_pct_b":   boll_df["boll_pct_b"],
        "esto_k":       esto_df["esto_k"],
        "esto_d":       esto_df["esto_d"],
    })

    # 4. Limpiar NaN e Infinity → convierte todo a None para JSON
    registros = limpiar_dataframe(resultado)

    # 5. Señales del último día
    ultimo = registros[-1] if registros else {}
    señales = generar_señales(ultimo)

    # 6. Resumen del estado actual
    cierre_actual = limpiar_valor(precios.iloc[-1])
    sma20_actual  = limpiar_valor(sma_20.iloc[-1])
    sma50_actual  = limpiar_valor(sma_50.iloc[-1])

    resumen = {
        "rsi_actual":      ultimo.get("rsi_14"),
        "precio_vs_sma20": "sobre" if (cierre_actual and sma20_actual and cierre_actual > sma20_actual) else "bajo",
        "precio_vs_sma50": "sobre" if (cierre_actual and sma50_actual and cierre_actual > sma50_actual) else "bajo",
        "macd_positivo":   (ultimo.get("macd") or 0) > 0,
        "boll_pct_b":      ultimo.get("boll_pct_b"),
    }

    return {
        "ticker":     ticker,
        "total_dias": len(registros),
        "señales":    señales,
        "resumen":    resumen,
        "datos":      registros,
    }
