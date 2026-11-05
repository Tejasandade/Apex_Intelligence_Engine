# ============================================================================
# Apex Intelligence Engine — Dockerfile
# For cloud deployment (Oracle Cloud, Railway, Fly.io, etc.)
# ============================================================================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer cache)
COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

# Copy the whole project
COPY . .

# Create log and state directories
RUN mkdir -p logs data/apex_state configs

# Default: headless server mode
ENV APEX_HEADLESS=true
ENV APEX_DEMO_MODE=false
ENV APEX_SYMBOL=BTCUSDT
ENV APEX_INITIAL_CAPITAL=1000.0
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Run the bot
CMD ["python", "scripts/apex_v7.py"]
