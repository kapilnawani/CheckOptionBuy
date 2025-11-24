# Dockerfile — Quick fallback
FROM mcr.microsoft.com/playwright/python:1.48.0-jammy

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ensure the installed playwright has the browsers
RUN python -m playwright install --with-deps chromium

ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
