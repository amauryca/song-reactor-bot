# Use slim Python base image
FROM python:3.11-slim

# Install ffmpeg (required by Whisper for MP3 decoding)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY reactor.py .

# Render passes the port via $PORT; default to 10000
EXPOSE 10000

# Single worker — Whisper model is large; multiple workers would exceed free RAM
CMD gunicorn --bind "0.0.0.0:${PORT:-10000}" --timeout 300 --workers 1 reactor:app
