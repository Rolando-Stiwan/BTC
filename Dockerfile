# Imagen base oficial de Python 3.12 en su versión ligera (slim)
FROM python:3.12-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia primero el archivo de dependencias para aprovechar la cache de Docker
COPY requirements.txt .

# Instala pip actualizado y las dependencias necesarias
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copia el resto del proyecto al contenedor
COPY . .

# Expone el puerto donde se ejecutará la aplicación FastAPI (por defecto 8000)
EXPOSE 8000

# Comando para ejecutar la aplicación con uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]