FROM python:3.11-slim

WORKDIR /app

# Install system deps (optional but helpful for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Python deps first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# App source
COPY . /app

# Cloud Run listens on $PORT (default 8080)
EXPOSE 8080
ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
