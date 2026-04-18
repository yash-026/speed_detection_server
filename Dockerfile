# ============ BUILDER STAGE ============
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies only in builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libopenblas-dev \
    liblapack-dev \
    libgfortran5 \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install all packages to /build/install
COPY requirements.txt .
RUN pip install --target /build/install --no-cache-dir -r requirements.txt && \
    find /build/install -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /build/install -type f -name "*.pyc" -delete && \
    find /build/install -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# ============ RUNTIME STAGE ============
FROM python:3.12-slim

WORKDIR /app

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    liblapack3 \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir gunicorn

# Copy installed packages from builder
COPY --from=builder /build/install /usr/local/lib/python3.12/site-packages

# Copy application code
COPY app.py .
COPY .env* ./

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Use gunicorn with optimized settings
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT:-5001}", "--workers", "1", "--threads", "4", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
