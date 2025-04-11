FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código de la aplicación
COPY . .

# Crear directorios necesarios
RUN mkdir -p uploads
RUN mkdir -p static

# Exponer el puerto que Railway usará
EXPOSE $PORT

# Comando para ejecutar la aplicación
CMD uvicorn app:app --host 0.0.0.0 --port $PORT