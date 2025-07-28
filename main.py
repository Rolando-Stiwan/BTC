from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

app = FastAPI()

# Cargar modelo entrenado
model = tf.keras.models.load_model("modelo_btc_lstm.h5")

# Input esperado: features + precio actual
class InputData(BaseModel):
    features: list[list[float]]  # Matriz: (window, n_features)
    current_price: float

@app.post("/predict")
def predict_price(data: InputData):
    input_array = np.array(data.features).reshape(1, len(data.features), len(data.features[0]))
    
    predicted_price = model.predict(input_array)[0][0]

    # Solo dos señales posibles
    signal = "BUY" if predicted_price > data.current_price else "SELL"

    return {
        "predicted_price": round(float(predicted_price), 2),
        "current_price": round(data.current_price, 2),
        "signal": signal
    }

@app.get("/healthz")
def health_check():
    return {"status": "ok"}