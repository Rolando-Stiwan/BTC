# Imagen base con Python 3.12 (versión slim para que sea más ligera)
FROM python:3.12-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia solo el archivo de dependencias primero (para aprovechar cache)
COPY requirements.txt .

# Actualiza pip e instala las dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copia todo el código del proyecto dentro del contenedor
COPY . .

# Expone el puerto donde correrá la app (8000 por defecto para FastAPI)
EXPOSE 8000

# Comando para ejecutar la app con uvicorn (ajusta si usas otro nombre)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
