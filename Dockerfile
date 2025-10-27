# =====================================
# Production-ready Dockerfile for API
# =====================================

FROM python:3.10-slim AS base

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src ./src
COPY models ./models
COPY params.yaml .

RUN mkdir -p data/raw data/processed mlruns

# Expose API port
EXPOSE 8000

# Healthcheck without requests dependency (curl present)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Default start command
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
