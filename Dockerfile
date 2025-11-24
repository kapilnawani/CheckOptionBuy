# Dockerfile
FROM mcr.microsoft.com/playwright/python:1.56.0-jammy

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy source
COPY . .

# ensure playwright browsers installed (image already has them but this ensures compatibility)
RUN python -m playwright install --with-deps chromium

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
