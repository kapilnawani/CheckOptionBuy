# Use official Playwright Python base image (includes Chromium + deps)
FROM mcr.microsoft.com/playwright/python:v1.56.0-jammy

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Expose port (Railway will map it)
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
