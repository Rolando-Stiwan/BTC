from fastapi import FastAPI
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from ta.momentum import RSIIndicator
import joblib

app = FastAPI()

# Cargar modelo y scaler
model = load_model("modelo_btc_lstm.h5", compile=False)
scaler = joblib.load("scaler.pkl")

@app.get("/predict")
def predict():
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

    # Escalar usando scaler entrenado
    df_scaled = scaler.transform(df_feat.tail(10))

    # Preparar entrada para el modelo LSTM
    input_array = df_scaled.reshape(1, 10, 6)

    # Predicción
    predicted_price = model.predict(input_array)[0][0]
    current_price = df["btc_close"].iloc[-1]

    # Señal: comparar retornos logarítmicos para decidir compra o venta
    signal = "BUY" if predicted_price > current_price else "SELL"

    return {
        "predicted_price": round(float(predicted_price), 6),
        "current_price": round(float(current_price), 6),
        "signal": signal
    }

@app.get("/healthz")
def health_check():
    return {"status": "ok"}