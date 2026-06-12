"""
Clinical AI Copilot — FastAPI entrypoint.

Endpoints
  GET  /health                     liveness + which providers are configured
  GET  /api/providers              LLM provider catalog (local + hosted)
  GET  /api/agents                 multi-agent roster (for the UI)
  GET  /api/samples                built-in demo datasets
  GET  /api/knowledge-graph        ICD-10/SNOMED/RxNorm graph for visualisation
  POST /api/analyze                multipart upload  → SSE multi-agent run
  POST /api/analyze/sample         JSON {dataset_id} → SSE multi-agent run
  POST /api/chat                   SSE token stream from the chat copilot
"""
from __future__ import annotations

import json
import os

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from . import knowledge, llm, samples
from .agents import AGENTS, RunContext, chat_stream, run_pipeline
from .eeg import load_signal
from .schemas import ChatRequest

app = FastAPI(title="Clinical AI Copilot", version="2.0.0",
              docs_url="/api/docs", redoc_url="/api/redoc")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CLINICAL_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                "Connection": "keep-alive"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "clinical-ai-copilot",
        "version": app.version,
        "llm_configured": bool(os.getenv("CLINICAL_LLM_API_KEY")
                               or os.getenv("CLINICAL_LLM_PROVIDER")),
        "agents": len(AGENTS),
    }


@app.get("/api/providers")
async def providers():
    return {"providers": llm.provider_catalog()}


@app.get("/api/agents")
async def agents():
    return {"agents": AGENTS}


@app.get("/api/samples")
async def list_samples():
    return {"samples": list(samples.SAMPLES.values())}


@app.get("/api/knowledge-graph")
async def kg():
    return knowledge.knowledge_graph()


def _parse_llm(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _parse_patient(raw: str | None) -> dict:
    base = {"patient_id": "UPLOAD", "symptoms": "", "medications": []}
    if not raw:
        return base
    try:
        data = json.loads(raw)
        if isinstance(data.get("medications"), str):
            data["medications"] = [m.strip() for m in data["medications"].split(",") if m.strip()]
        return {**base, **data}
    except json.JSONDecodeError:
        return base


@app.post("/api/analyze")
async def analyze(
    file: UploadFile | None = File(default=None),
    patient: str | None = Form(default=None),
    llm_config: str | None = Form(default=None),
):
    patient_d = _parse_patient(patient)
    llm_d = _parse_llm(llm_config)

    if file is not None:
        raw = await file.read()
        signal, fs, meta = load_signal(file.filename or "upload.csv", raw)
    else:  # no file → synthetic ictal demo
        signal = samples.synth("ictal")
        from .eeg import DEFAULT_FS
        fs, meta = DEFAULT_FS, {"source": "synthetic-ictal", "channels": 16}

    ctx = RunContext(patient=patient_d, llm_override=llm_d, signal=signal, fs=fs, meta=meta)
    return StreamingResponse(run_pipeline(ctx), media_type="text/event-stream",
                             headers=_SSE_HEADERS)


@app.post("/api/analyze/sample")
async def analyze_sample(body: dict):
    dataset_id = body.get("dataset_id", "ictal")
    spec = samples.SAMPLES.get(dataset_id, samples.SAMPLES["ictal"])
    signal = samples.synth(dataset_id)
    from .eeg import DEFAULT_FS
    patient = {**spec["patient"], **(body.get("patient") or {})}
    ctx = RunContext(patient=patient, llm_override=body.get("llm", {}),
                     signal=signal, fs=DEFAULT_FS,
                     meta={"source": f"sample:{dataset_id}", "channels": 16})
    return StreamingResponse(run_pipeline(ctx), media_type="text/event-stream",
                             headers=_SSE_HEADERS)


@app.post("/api/chat")
async def chat(req: ChatRequest):
    context = {
        "analysis": req.analysis,
        "patient": req.patient.model_dump() if req.patient else None,
    }
    messages = [m.model_dump() for m in req.messages]
    return StreamingResponse(
        chat_stream(messages, context, req.llm.model_dump()),
        media_type="text/event-stream", headers=_SSE_HEADERS,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
