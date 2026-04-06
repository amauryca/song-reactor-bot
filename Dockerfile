# Use slim Python base image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TMPDIR=/tmp \
    WHISPER_CACHE_DIR=/tmp/whisper

# Install runtime system packages required by Whisper and audio decoding.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

# Copy app code
COPY reactor.py .

# Render passes the port via $PORT; default to 10000
EXPOSE 10000

# Single worker — Whisper model is large; multiple workers would exceed free RAM
CMD gunicorn --bind "0.0.0.0:${PORT:-10000}" --workers 1 --threads 2 --timeout 180 --graceful-timeout 30 --keep-alive 5 --max-requests 50 --max-requests-jitter 10 reactor:app
