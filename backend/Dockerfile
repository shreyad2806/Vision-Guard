# ---------------------------------------------------------------------------
# VisionGuard Backend — Dockerfile
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for opencv-python-headless
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY apps/ apps/

# Copy ML model artifacts
COPY models/ models/

# Create data directories
RUN mkdir -p data/uploads

EXPOSE 8000

CMD ["uvicorn", "apps.main:app", "--host", "0.0.0.0", "--port", "8000"]
