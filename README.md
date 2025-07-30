# 🔮 API de Predicción BTC LSTM (FastAPI)

Este repositorio contiene una API desarrollada con **FastAPI** para predecir señales de trading (BUY/SELL) de Bitcoin (`BTC-USD`) mediante una red neuronal LSTM previamente entrenada. La API está desplegada en https://btc-6vq9.onrender.com/docs

---

## 🚀 Descripción del proyecto

La API recibe una consulta HTTP y devuelve señales de compra o venta de BTC, basadas en una secuencia de características extraídas de datos recientes de Bitcoin y otros features. Se utiliza un modelo LSTM y un `scaler.pkl` previamente entrenado.

---

## 🧠 Modelo LSTM

El modelo fue entrenado previamente con una arquitectura LSTM secuencial para predecir el retorno logarítmico diario del precio de BTC. Luego, ese retorno se convierte en una predicción de precio y se compara con el precio anterior para emitir la señal BUY o SELL.

---

## 📡 Endpoint principal

`GET /predict`

### Parámetros:
- `days` (int): Número de días a predecir (entre 1 y 7). Valor por defecto: 1.

### Respuesta:
```json
{
  "current_price": 63500.21,
  "signals": ["BUY", "BUY", "SELL"]
} 
```

## ⚙️ Requisitos
Instala las dependencias:
```bash
pip install -r requirements.txt
```

## 🤝 Créditos
Desarrollado por Rolando Stiwan — Científico de Datos con enfoque en finanzas cuantitativas, deep learning y riesgo.

## ⚠️ Disclaimer
Este proyecto tiene fines exclusivamente educativos. Las señales generadas no deben ser consideradas recomendaciones de inversión.