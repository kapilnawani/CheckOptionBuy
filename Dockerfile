FROM python:3.10-slim

# Prevent Python from writing .pyc files & enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System updates
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers + OS deps for Chromium
RUN python -m playwright install --with-deps chromium

# Copy app code
COPY . .

# Railway provides PORT env; default to 8000 for local runs
ENV PORT=8000

# Start FastAPI (api_server.py) with uvicorn
CMD ["sh", "-c", "uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8000}"]
