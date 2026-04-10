# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# system deps (if catboost needs build tools; often not needed for wheel)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# copy requirements & install (cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app and models
COPY app.py .
COPY models/ ./models/

EXPOSE 8000

# run with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
