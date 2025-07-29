from fastapi import FastAPI, Query
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from ta.momentum import RSIIndicator
import joblib

app = FastAPI()

model = load_model("modelo_btc_lstm.h5", compile=False)
scaler = joblib.load("scaler.pkl")

@app.get("/predict")
def predict(days: int = Query(1, ge=1, le=7)):
    # Descargar datos de los últimos 40 días
    btc = yf.download("BTC-USD", period="40d", interval="1d")
    sp500 = yf.download("^GSPC", period="40d", interval="1d")
    nasdaq = yf.download("^IXIC", period="40d", interval="1d")

    if btc.empty or sp500.empty or nasdaq.empty:
        return {"error": "Error al obtener datos del mercado."}

    # Sincronizar fechas entre activos (quedarnos solo con fechas comunes)
    common_dates = btc.index.intersection(sp500.index).intersection(nasdaq.index)
    btc = btc.loc[common_dates]
    sp500 = sp500.loc[common_dates]
    nasdaq = nasdaq.loc[common_dates]

    # Crear DataFrame conjunto con fechas comunes
    df = pd.DataFrame(index=common_dates)
    df["btc_close"] = btc["Close"]
    df["btc_volume"] = btc["Volume"]
    df["sp500_close"] = sp500["Close"]
    df["nasdaq_close"] = nasdaq["Close"]

    # Calcular features
    df["return_btc_price"] = np.log(df["btc_close"] / df["btc_close"].shift(1))
    df["return_sp500"] = np.log(df["sp500_close"] / df["sp500_close"].shift(1))
    df["return_Nasdaq"] = np.log(df["nasdaq_close"] / df["nasdaq_close"].shift(1))
    df["diff_volume"] = df["btc_volume"].diff()
    df["sma5"] = df["btc_close"].rolling(window=5).mean()
    df["diff_sma5"] = df["sma5"].diff()
    df["rsi"] = RSIIndicator(close=df["btc_close"], window=14).rsi()

    # Limpiar y seleccionar features
    df_feat = df[[
        "return_btc_price", "rsi", "diff_volume",
        "diff_sma5", "return_sp500", "return_Nasdaq"
    ]].dropna()

    # Verificar que haya suficientes datos (al menos 10 para la ventana)
    if df_feat.shape[0] < 10:
        return {"error": "No hay suficientes datos válidos para generar una predicción."}

    # Preparar la secuencia inicial para predecir
    input_seq = df_feat.tail(10).copy().values
    input_scaled = scaler.transform(input_seq)
    input_scaled = input_scaled.reshape(1, 10, 6)

    current_price = df["btc_close"].iloc[-1]
    last_features = df_feat.tail(1).values[0]

    signals = []

    for _ in range(days):
        # Predecir retorno logarítmico escalado
        pred_scaled = model.predict(input_scaled)[0][0]

        # Desescalar la predicción para obtener retorno real
        dummy_input = np.zeros((1, 6))
        dummy_input[0, 0] = pred_scaled
        pred_return = scaler.inverse_transform(dummy_input)[0, 0]

        # Convertir retorno a precio predicho real
        pred_price = current_price * np.exp(pred_return)

        # Señal comparando con el precio actual (o predicho del paso anterior)
        signal = "BUY" if pred_price > current_price else "SELL"
        signals.append(signal)

        # Actualizar para la siguiente iteración
        current_price = pred_price

        # Crear nueva fila de features para el próximo input_scaled
        new_features = last_features.copy()
        new_features[0] = pred_return  # return_btc_price es la primera feature
        # Aquí podrías actualizar otras features si tienes cómo, pero las mantenemos iguales
        last_features = new_features

        # Transformar y actualizar input_scaled
        new_scaled = scaler.transform([new_features])[0]
        input_scaled = np.append(input_scaled[:, 1:, :], [[new_scaled]], axis=1)

    return {
        "current_price": round(float(df["btc_close"].iloc[-1]), 2),
        "signals": signals
    }