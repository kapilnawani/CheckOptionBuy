# Dockerfile — Reproducible slim-based build (recommended if image tags fail)
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies required by Playwright browsers
RUN apt-get update && apt-get install -y \
    wget curl ca-certificates gnupg libnss3 libxss1 libasound2 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libxcomposite1 libxdamage1 libxrandr2 libgbm1 libpango-1.0-0 libpangocairo-1.0-0 libx11-xcb1 libxtst6 \
    libxkbfile1 libglib2.0-0 tzdata --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Install Python deps including Playwright. Fix the playwright version here.
RUN pip install --no-cache-dir -r requirements.txt

# Install browsers for Playwright (with deps)
RUN python -m playwright install --with-deps chromium

# Copy app
COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
