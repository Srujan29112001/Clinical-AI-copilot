# ── Clinical AI Copilot — backend (FastAPI multi-agent service) ──
# Build from the REPO ROOT so the ontology data is in the build context:
#   docker build -f backend/Dockerfile -t clinical-ai-backend .
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CLINICAL_CORS_ORIGINS=* \
    CLINICAL_ONTOLOGY_DIR=/app/data/medical_ontology

COPY backend/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY backend/app ./app
COPY data/medical_ontology ./data/medical_ontology

# Honour $PORT (Render, Railway, Hugging Face Spaces all inject it; HF uses 7860).
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
