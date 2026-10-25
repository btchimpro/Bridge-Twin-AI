FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for PostgreSQL/PostGIS client and compilation
RUN apt-get update && apt-get install -y libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from backend directory
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application source code
COPY backend/ /app/

# Copy trained AI pipeline models into container
COPY ai_pipeline/ /ai_pipeline/

ENV PYTHONUNBUFFERED=1
ENV AI_MODELS_DIR=/ai_pipeline/models
ENV PORT=8000

EXPOSE 8000

# Use Render-assigned PORT dynamic variable or default to 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
