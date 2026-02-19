FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source + faq.json into the container image
COPY . .

# Cloud Run provides PORT at runtime; default to 8080 if not set
ENV PORT=8080
EXPOSE 8080

# Use shell form so $PORT expands
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT}"
