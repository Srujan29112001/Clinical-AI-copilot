# Hugging Face Spaces — Clinical AI Copilot backend (Docker SDK, free CPU tier).
# This single file is all the Space needs: it pulls the latest backend code from
# GitHub at build time, keeping backend/ + data/ in their original layout so the
# app's relative paths to data/medical_ontology and data/samples just work.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CLINICAL_CORS_ORIGINS=*

# git to fetch the repo; libgomp1 for SciPy/NumPy runtime.
RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --depth 1 https://github.com/Srujan29112001/Clinical-AI-copilot.git .

RUN pip install --upgrade pip && pip install -r backend/requirements.txt

# Run from backend/ so app.main and the data/ paths resolve as in the repo.
WORKDIR /app/backend
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
