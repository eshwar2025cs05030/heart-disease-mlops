# ============================================================
# Dockerfile — Heart Disease Prediction API
# MLOps Assignment: BITS Pilani MTech (AMLCSZG523)
# ============================================================

# Stage 1: Builder — install dependencies
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ============================================================
# Stage 2: Runtime — lean final image
FROM python:3.10-slim AS runtime

LABEL maintainer="Eshwar Pradeep K <eshwarpradeep.k95@gmail.com>"
LABEL version="1.0.0"
LABEL description="Heart Disease Prediction API — BITS Pilani MTech MLOps Assignment"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/app.py .
COPY src/data_preprocessing.py .
COPY models/ ./models/

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose API port
ENV MODEL_PATH=/app/models/best_model.pkl
ENV PIPELINE_PATH=/app/models/preprocessing_pipeline.pkl

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
