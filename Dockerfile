# Production Dockerfile for Insurance Charges Prediction API Service
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install curl for container health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first for efficient Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy project package configuration and source code
COPY pyproject.toml README.md .flake8 ./
COPY config/ config/
COPY models/ models/
COPY reports/ reports/
COPY src/ src/

# Install the package in editable mode without build isolation
RUN pip install --no-cache-dir --no-build-isolation -e .

# Create non-root system user for security compliance (Principle of Least Privilege)
RUN useradd -m -u 10001 -s /bin/bash appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Container healthcheck using standard endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn ASGI server
CMD ["uvicorn", "insurance_mlops.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
