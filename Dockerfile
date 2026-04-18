FROM python:3.12-slim

WORKDIR /app

# Install minimal system dependencies needed for build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY .env* ./

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Use gunicorn to run the app
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT:-5001}", "--workers", "2", "--timeout", "60", "app:app"]
