"""
Multi-agent clinical pipeline.

Seven cooperating agents, each able to run on its own LLM provider (local GPU
or hosted API). The orchestrator streams structured progress events so the UI
can visualise the pipeline in real time (the Helix "Studio" pattern).

    Triage ─▶ SignalAnalyst ─▶ KnowledgeRetriever ─▶ Diagnostician
                                                          │
                                  Pharmacologist ◀────────┤
                                                          ▼
                                   SafetyCritic ─▶ Reporter
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator

import numpy as np

from . import eeg, knowledge
from .llm import get_llm
from .schemas import AgentEvent

AGENTS = [
    {"id": "triage", "name": "Triage Agent", "role": "Acuity & urgency assessment",
     "icon": "siren"},
    {"id": "signal", "name": "Signal Analyst", "role": "EEG preprocessing & feature extraction",
     "icon": "activity"},
    {"id": "retriever", "name": "Knowledge Retriever", "role": "GraphRAG over ICD-10 / SNOMED / RxNorm",
     "icon": "network"},
    {"id": "diagnostician", "name": "Diagnostician", "role": "Differential diagnosis reasoning",
     "icon": "stethoscope"},
    {"id": "pharmacologist", "name": "Pharmacologist", "role": "Drug-drug interaction safety",
     "icon": "pill"},
    {"id": "critic", "name": "Safety Critic", "role": "Self-correction & guardrails",
     "icon": "shield"},
    {"id": "reporter", "name": "Reporter", "role": "Clinical report synthesis",
     "icon": "file-text"},
]


def _ev(**kw) -> str:
    return "data: " + AgentEvent(**kw).model_dump_json(exclude_none=True) + "\n\n"


@dataclass
class RunContext:
    patient: dict
    llm_override: dict
    signal: np.ndarray
    fs: float
    meta: dict


async def run_pipeline(ctx: RunContext) -> AsyncIterator[str]:
    """Yield SSE strings as the pipeline progresses."""
    patient = ctx.patient
    symptoms = patient.get("symptoms", "")
    meds = patient.get("medications", []) or []
    result: dict[str, Any] = {"patient_id": patient.get("patient_id", "DEMO")}

    yield _ev(type="log", message="Pipeline initialised — 7 agents online")

    # 1 ── Signal Analyst ----------------------------------------------------
    async for e in _stage("signal", "Filtering, artefact handling & feature extraction"):
        yield e
    features = await asyncio.to_thread(eeg.analyze_array, ctx.signal, ctx.fs)
    result["eeg"] = {**features, **{"recording": ctx.meta}}
    yield _ev(type="log", agent="signal",
              message=f"Extracted {len(features['band_powers'])} bands · "
                      f"seizure p={features['seizure_probability']:.2f}")
    yield _ev(type="stage", agent="signal", status="done",
              data={"band_powers": features["band_powers"],
                    "seizure_probability": features["seizure_probability"]})

    # 2 ── Triage ------------------------------------------------------------
    async for e in _stage("triage", "Assessing clinical urgency"):
        yield e
    prob = features["seizure_probability"]
    llm, cfg = get_llm("triage", ctx.llm_override)
    triage_prompt = (
        f"seizure_probability: {prob}\nsymptoms: {symptoms}\n"
        f"reasons: {'; '.join(features['seizure_reasons'])}\n"
        "Give a one-paragraph triage decision with an explicit URGENCY level."
    )
    triage_text = await llm.complete(_SYS["triage"], triage_prompt)
    urgency = ("Emergent" if prob >= 0.8 else "Urgent" if prob >= 0.45 else "Routine")
    result["triage"] = {"urgency": urgency, "narrative": triage_text, "provider": cfg.provider}
    yield _ev(type="stage", agent="triage", status="done",
              message=triage_text, data={"urgency": urgency})

    # 3 ── Knowledge Retriever ----------------------------------------------
    async for e in _stage("retriever", "Retrieving evidence from the knowledge graph"):
        yield e
    rag = await asyncio.to_thread(knowledge.retrieve, features, symptoms)
    result["diagnoses"] = rag["diagnoses"]
    result["treatments"] = rag["treatments"]
    result["evidence"] = rag["evidence"]
    top = rag["diagnoses"][0] if rag["diagnoses"] else {}
    yield _ev(type="stage", agent="retriever", status="done",
              message=f"{len(rag['diagnoses'])} candidate diagnoses · top: "
                      f"{top.get('condition', 'n/a')[:60]}",
              data={"diagnoses": rag["diagnoses"][:3]})

    # 4 ── Diagnostician -----------------------------------------------------
    async for e in _stage("diagnostician", "Reasoning over fused EEG + clinical evidence"):
        yield e
    llm, cfg = get_llm("diagnostician", ctx.llm_override)
    dx_prompt = (
        f"top_diagnosis: {top.get('condition', 'unknown')}\n"
        f"icd10: {top.get('icd10', '')}\n"
        f"eeg_markers: {', '.join(top.get('eeg_markers', [])) or features['dominant_band'] + ' dominant'}\n"
        f"symptoms: {symptoms}\nseizure_probability: {prob}\n"
        "Explain the leading diagnosis and key differentials in one paragraph."
    )
    dx_text = await llm.complete(_SYS["diagnostician"], dx_prompt)
    severity = top.get("severity", "moderate")
    confidence = top.get("confidence", 0.6)
    result["primary_diagnosis"] = {
        "condition": top.get("condition", "Undetermined"),
        "icd10": top.get("icd10"), "confidence": confidence,
        "reasoning": dx_text, "provider": cfg.provider,
    }
    result["severity"] = severity
    result["urgency"] = urgency
    yield _ev(type="stage", agent="diagnostician", status="done",
              message=dx_text,
              data={"condition": top.get("condition"), "confidence": confidence})

    # 5 ── Pharmacologist ----------------------------------------------------
    async for e in _stage("pharmacologist", "Checking drug-drug interactions"):
        yield e
    di = await asyncio.to_thread(knowledge.drug_interactions, meds)
    result["drug_safety"] = di
    msg = (f"{di['count']} interaction(s); highest: {di['highest_severity']}"
           if di["count"] else "No significant interactions detected")
    yield _ev(type="stage", agent="pharmacologist", status="done",
              message=msg, data=di)

    # 6 ── Safety Critic (self-correction / guardrails) ----------------------
    async for e in _stage("critic", "Running guardrails & calibration check"):
        yield e
    flags = _safety_review(result)
    result["safety"] = flags
    yield _ev(type="stage", agent="critic", status="done",
              message="; ".join(flags["notes"]) or "All checks passed",
              data=flags)

    # 7 ── Reporter ----------------------------------------------------------
    async for e in _stage("reporter", "Synthesising the clinical report"):
        yield e
    llm, cfg = get_llm("reporter", ctx.llm_override)
    rep_prompt = (
        f"top_diagnosis: {top.get('condition', 'unknown')}\n"
        f"severity: {severity}\nurgency: {urgency}\n"
        f"seizure_probability: {prob}\n"
        "Write a concise clinical summary suitable for the chart."
    )
    rep_text = await llm.complete(_SYS["reporter"], rep_prompt)
    result["summary"] = rep_text
    result["report_id"] = f"RPT-{abs(hash(str(result))) % 10**8:08d}"
    result["disclaimer"] = (
        "Decision-support only. Not FDA-approved. Requires review by a qualified clinician."
    )
    yield _ev(type="stage", agent="reporter", status="done", message=rep_text)

    yield _ev(type="result", data=result)
    yield _ev(type="done", message="Analysis complete")


async def _stage(agent: str, message: str) -> AsyncIterator[str]:
    yield _ev(type="stage", agent=agent, status="running", message=message)
    await asyncio.sleep(0.25)  # let the UI render the running state


def _safety_review(result: dict) -> dict:
    notes = []
    conf = result.get("primary_diagnosis", {}).get("confidence", 0)
    if conf < 0.6:
        notes.append("Low diagnostic confidence — recommend specialist confirmation")
    if result.get("drug_safety", {}).get("requires_action"):
        notes.append("Major/contraindicated drug interaction flagged — review before prescribing")
    if result.get("eeg", {}).get("seizure_probability", 0) >= 0.8:
        notes.append("High seizure probability — time-critical pathway")
    if result.get("eeg", {}).get("recording", {}).get("load_error"):
        notes.append("Uploaded file could not be fully parsed — synthetic fallback used")
    return {"passed": not notes, "notes": notes,
            "calibrated_confidence": round(min(conf, 0.95), 3)}


_SYS = {
    "triage": "You are a neuro-ICU triage clinician. Be decisive and concise. "
              "Always state an explicit urgency level (Routine/Urgent/Emergent).",
    "diagnostician": "You are a board-certified neurologist reasoning over EEG and "
                     "clinical data. Be precise, hedge appropriately, name differentials.",
    "reporter": "You are a clinical scribe. Produce a tight, structured chart summary. "
                "No fabricated values; rely only on the provided context.",
}


# ── chat (conversational copilot over a completed analysis) ─────────────────
async def chat_stream(messages: list[dict], context: dict, llm_override: dict) -> AsyncIterator[str]:
    llm, cfg = get_llm("chat", llm_override)
    last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    ctx_lines = []
    if context.get("analysis"):
        a = context["analysis"]
        pd = a.get("primary_diagnosis", {})
        ctx_lines.append(f"current_diagnosis: {pd.get('condition')} ({pd.get('icd10')})")
        ctx_lines.append(f"seizure_probability: {a.get('eeg', {}).get('seizure_probability')}")
        ctx_lines.append(f"urgency: {a.get('urgency')}")
    if context.get("patient"):
        ctx_lines.append(f"patient_symptoms: {context['patient'].get('symptoms')}")
    prompt = "\n".join(ctx_lines) + f"\nquestion: {last}"
    yield _ev(type="stage", agent="chat", status="running")
    async for tok in llm.stream(_CHAT_SYS, prompt):
        yield _ev(type="token", message=tok)
    yield _ev(type="done")


_CHAT_SYS = (
    "You are the Clinical AI Copilot assistant. You help clinicians interpret EEG "
    "analyses, diagnoses, drug-safety findings and the knowledge graph. Be accurate, "
    "cite the structured context, and never invent patient data. Add a brief safety "
    "note when discussing treatment."
)
