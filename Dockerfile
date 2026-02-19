# Dockerfile (Cloud Run)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy and install deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code (includes faq.json)
COPY . /app

# Cloud Run sets PORT; default to 8080
ENV PORT=8080
EXPOSE 8080

# IMPORTANT: Bind to 0.0.0.0 and use $PORT
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
