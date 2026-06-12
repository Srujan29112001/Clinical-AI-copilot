# Clinical AI Copilot — Backend

FastAPI service that runs the **seven-agent clinical pipeline** with a **hybrid
LLM layer** (local GPU *or* any API provider) and streams progress over SSE.

## Run locally

```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate   |   macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/api/docs for the interactive Swagger UI.

## Inference modes

Everything works with **zero configuration** (deterministic demo reasoning).
To use a real model, set environment variables (see `.env.example`):

| Mode | Variables |
|---|---|
| **Local GPU** (Ollama) | `CLINICAL_LLM_PROVIDER=ollama` · `CLINICAL_LLM_BASE_URL=http://localhost:11434/v1` · `CLINICAL_LLM_MODEL=llama3.1:8b` |
| **Local GPU** (vLLM) | `CLINICAL_LLM_PROVIDER=vllm` · `CLINICAL_LLM_BASE_URL=http://localhost:8001/v1` |
| **Anthropic** | `CLINICAL_LLM_PROVIDER=anthropic` · `CLINICAL_LLM_API_KEY=sk-ant-…` |
| **OpenAI / Groq / DeepSeek / Mistral / Gemini / OpenRouter** | `CLINICAL_LLM_PROVIDER=<id>` · `CLINICAL_LLM_API_KEY=…` |

Per-agent overrides use `CLINICAL_<ROLE>_*` (roles: `TRIAGE`, `DIAGNOSTICIAN`,
`REPORTER`, `CHAT`) — e.g. run triage locally and diagnosis on Claude. The UI
can also pass a per-request config that overrides the environment.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/analyze` | multipart EEG upload → SSE multi-agent run |
| POST | `/api/analyze/sample` | `{dataset_id}` → SSE run on a built-in sample |
| POST | `/api/chat` | SSE token stream from the chat copilot |
| GET | `/api/knowledge-graph` | ICD-10 / SNOMED / RxNorm graph |
| GET | `/api/providers` · `/api/agents` · `/api/samples` | metadata for the UI |
| GET | `/health` | liveness + provider status |

## Supported EEG formats

`.csv` · `.tsv` · `.txt` · `.npy` · `.json` (channels × samples). `.edf`/`.bdf`
require the optional `pyedflib` extra. Unparseable files fall back to a
synthetic signal so the demo never breaks.

## Docker

```bash
# from the repo root (data/ must be in the build context)
docker build -f backend/Dockerfile -t clinical-ai-backend .
docker run -p 8000:8000 clinical-ai-backend
```
