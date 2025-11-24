# Dockerfile — slim-based reproducible build for Python 3.10
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system deps required by Playwright browsers + some useful tools
RUN apt-get update && apt-get install -y \
    wget curl ca-certificates gnupg libnss3 libxss1 libasound2 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libxcomposite1 libxdamage1 libxrandr2 libgbm1 libpango-1.0-0 libpangocairo-1.0-0 libx11-xcb1 libxtst6 \
    libxkbfile1 libglib2.0-0 tzdata build-essential git --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and upgrade pip first (helps pick compatible wheels)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Install Playwright browsers (chromium). This must match the playwright pip version.
RUN python -m playwright install --with-deps chromium

# Copy app code
COPY . /app

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

# Use single worker to make logs easier to follow on Railway; increase later if needed.
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
