"""
Multi-agent clinical pipeline (v3).

Eleven cooperating agents, each able to run on its own LLM provider/model
(local GPU or hosted API). The pipeline now drives a real qEEG feature engine,
a neuromorphic SNN (with spike raster), a Mamba2 temporal model, a 7-way
multi-detector, and a multimodal cross-attention fusion head — streaming
structured progress so the UI can visualise everything in real time.

  Signal Analyst ─▶ Neuromorphic(SNN) ─▶ Multi-Detector ─▶ Temporal(Mamba2)
        │                                                        │
        └───────────────▶ Triage ◀───────────────────────────────┘
                            │
        Knowledge Retriever (GraphRAG + literature)
                            │
        Multimodal Fusion ─▶ Diagnostician ─▶ Pharmacologist ─▶ Safety Critic ─▶ Reporter
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator

import numpy as np

from . import eeg, features, detectors, knowledge
from .models_np import (
    SpikingNeuralNetwork, SelectiveSSM, CNNLSTMEncoder, TextEncoder,
    MultimodalFusion, event_evidence, SEVERITY, URGENCY,
)
from .llm import get_llm
from .schemas import AgentEvent

AGENTS = [
    {"id": "signal", "name": "Signal Analyst", "role": "qEEG feature extraction (50+ params) + CNN-LSTM embedding", "icon": "activity", "model": "CNN-LSTM"},
    {"id": "neuro", "name": "Neuromorphic Detector", "role": "Spiking neural network event detection (LIF, spike raster)", "icon": "zap", "model": "Spiking NN"},
    {"id": "detect", "name": "Multi-Detector", "role": "7 parallel detectors: seizure, sleep, encephalopathy, burst-suppression…", "icon": "scan", "model": "qEEG rules"},
    {"id": "temporal", "name": "Temporal Modeler", "role": "Mamba2 selective state-space trend over analysis windows", "icon": "waves", "model": "Mamba2 SSM"},
    {"id": "triage", "name": "Triage Agent", "role": "Acuity & urgency assessment", "icon": "siren", "model": "LLM"},
    {"id": "retriever", "name": "Knowledge Retriever", "role": "GraphRAG over ICD-10/SNOMED/RxNorm + literature", "icon": "network", "model": "GraphRAG"},
    {"id": "fusion", "name": "Multimodal Fusion", "role": "Cross-attention EEG↔text → severity/urgency + MC-dropout uncertainty", "icon": "git-merge", "model": "Cross-attention"},
    {"id": "diagnostician", "name": "Diagnostician", "role": "Differential diagnosis reasoning", "icon": "stethoscope", "model": "LLM"},
    {"id": "pharmacologist", "name": "Pharmacologist", "role": "Drug-drug interaction safety", "icon": "pill", "model": "rules"},
    {"id": "critic", "name": "Safety Critic", "role": "Self-correction, calibration & guardrails", "icon": "shield", "model": "rules"},
    {"id": "reporter", "name": "Reporter", "role": "Clinical report synthesis", "icon": "file-text", "model": "LLM"},
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


def _window_sequence(x: np.ndarray, fs: float, n_windows: int = 8) -> np.ndarray:
    """Per-window 8-D feature vectors (5 rel band powers + LL + sync + HF) for the SSM."""
    from .features import BANDS, band_power, _welch
    segs = np.array_split(x, n_windows, axis=-1)
    seq = []
    for s in segs:
        if s.shape[-1] < 8:
            continue
        f, p = _welch(s, fs)
        pm = p.mean(axis=0); tot = np.trapz(pm, f) + 1e-12
        rel = [float(band_power(f, pm, lo, hi) / tot) for lo, hi in BANDS.values()]
        ll = float(np.abs(np.diff(s, axis=-1)).sum(axis=-1).mean()) / 5000
        sync = float(np.abs(np.corrcoef(s)[np.triu_indices(s.shape[0], 1)]).mean()) if s.shape[0] > 1 else 0
        hf = rel[3] + rel[4]
        seq.append(rel + [min(ll, 2), sync, hf])
    return np.array(seq) if seq else np.zeros((1, 8))


async def run_pipeline(ctx: RunContext) -> AsyncIterator[str]:
    patient = ctx.patient
    symptoms = patient.get("symptoms", "")
    meds = patient.get("medications", []) or []
    x, fs = ctx.signal, ctx.fs
    result: dict[str, Any] = {"patient_id": patient.get("patient_id", "DEMO"),
                              "models_used": {}}

    yield _ev(type="log", message="Pipeline initialised — 11 agents online")
    xf = await asyncio.to_thread(eeg.preprocess, x, fs)

    # 1 ── Signal Analyst: full qEEG + CNN-LSTM embedding ---------------------
    async for e in _stage("signal", "Extracting 50+ qEEG parameters & CNN-LSTM embedding"):
        yield e
    feats = await asyncio.to_thread(features.extract_all, x, fs)
    cnn = CNNLSTMEncoder()
    eeg_emb = await asyncio.to_thread(cnn.encode, xf, fs)
    result["qeeg"] = feats
    result["models_used"]["signal"] = "CNN-LSTM encoder"
    yield _ev(type="stage", agent="signal", status="done",
              message=f"{feats['n_features']} qEEG parameters across {len(feats['channels'])} channels; "
                      f"dominant {max(feats['spectral']['rel_band_power'], key=feats['spectral']['rel_band_power'].get)}",
              data={"n_features": feats["n_features"],
                    "rel_band_power": feats["spectral"]["rel_band_power"],
                    "connectivity": feats["connectivity"]})

    # 3 ── Multi-Detector (compute early; informs SNN + triage) ---------------
    det = await asyncio.to_thread(detectors.detect_all, x, fs, feats)
    sleep_stage = next((d["detail"]["stage"] for d in det["detections"] if d["id"] == "sleep"), "Wake")
    sz_prob = det["seizure_probability"]

    # 2 ── Neuromorphic Detector (SNN) ---------------------------------------
    async for e in _stage("neuro", "Running spiking neural network (LIF) event detection"):
        yield e
    snn = SpikingNeuralNetwork()
    evid = event_evidence(feats, sz_prob, sleep_stage)
    snn_out = await asyncio.to_thread(snn.forward, evid)
    result["neuromorphic"] = snn_out
    result["models_used"]["events"] = "Spiking NN (LIF)"
    yield _ev(type="stage", agent="neuro", status="done",
              message=f"Detected event: {snn_out['detected_event']} · firing rate {snn_out['firing_rate']:.2f} · "
                      f"{snn_out['total_spikes']} spikes (~{snn_out['energy_uj']} µJ)",
              data={"detected_event": snn_out["detected_event"],
                    "raster": snn_out["raster"], "raster_labels": snn_out["raster_labels"],
                    "firing_rate": snn_out["firing_rate"]})

    # 3 ── (emit) Multi-Detector ---------------------------------------------
    async for e in _stage("detect", "Scanning 7 detection use-cases"):
        yield e
    result["detections"] = det["detections"]
    result["primary_finding"] = det["primary_finding"]
    present = [d["label"] for d in det["detections"] if d["present"]]
    yield _ev(type="stage", agent="detect", status="done",
              message=f"Primary: {det['primary_finding']} · {len(present)} finding(s): {', '.join(present[:3]) or 'none'}",
              data={"detections": det["detections"], "primary_finding": det["primary_finding"]})

    # 4 ── Temporal Modeler (Mamba2) -----------------------------------------
    async for e in _stage("temporal", "Modelling temporal trend with Mamba2 selective state-space"):
        yield e
    seq = await asyncio.to_thread(_window_sequence, xf, fs)
    ssm = SelectiveSSM(d_model=seq.shape[1])
    temporal = await asyncio.to_thread(ssm.scan, seq)
    temporal.pop("embedding", None)
    result["temporal"] = temporal
    result["models_used"]["temporal"] = "Mamba2 SSM"
    yield _ev(type="stage", agent="temporal", status="done",
              message=f"Trend: {temporal['trend']} (slope {temporal['trend_slope']:+.3f}) over {temporal['windows']} windows",
              data=temporal)

    # 5 ── Triage ------------------------------------------------------------
    async for e in _stage("triage", "Assessing clinical urgency"):
        yield e
    urgency = ("Emergent" if sz_prob >= 0.8 or det["primary_finding"] == "burst_suppression"
               else "Urgent" if sz_prob >= 0.45 or det["primary_finding"] in ("encephalopathy", "focal")
               else "Routine")
    llm, cfg = get_llm("triage", ctx.llm_override)
    triage_text = await llm.complete(_SYS["triage"],
        f"seizure_probability: {sz_prob}\nprimary_finding: {det['primary_finding']}\n"
        f"symptoms: {symptoms}\ntrend: {temporal['trend']}\n"
        "Give a one-paragraph triage decision with an explicit URGENCY level.")
    result["triage"] = {"urgency": urgency, "narrative": triage_text, "provider": cfg.provider, "model": cfg.model}
    yield _ev(type="stage", agent="triage", status="done", message=triage_text, data={"urgency": urgency})

    # 6 ── Knowledge Retriever (GraphRAG + literature) -----------------------
    async for e in _stage("retriever", "Retrieving evidence from knowledge graph + literature"):
        yield e
    rag = await asyncio.to_thread(knowledge.retrieve, {"seizure_probability": sz_prob,
          "dominant_band": max(feats["spectral"]["rel_band_power"], key=feats["spectral"]["rel_band_power"].get)}, symptoms)
    lit = await asyncio.to_thread(knowledge.retrieve_literature,
          f"{symptoms} {det['primary_finding']} {snn_out['detected_event']}")
    result["diagnoses"] = rag["diagnoses"]
    result["treatments"] = rag["treatments"]
    result["evidence"] = rag["evidence"] + lit
    result["models_used"]["knowledge"] = "GraphRAG + literature"
    top = rag["diagnoses"][0] if rag["diagnoses"] else {}
    yield _ev(type="stage", agent="retriever", status="done",
              message=f"{len(rag['diagnoses'])} diagnoses + {len(lit)} papers · top: {top.get('condition','n/a')[:50]}",
              data={"diagnoses": rag["diagnoses"][:3], "literature": lit})

    # 7 ── Multimodal Fusion -------------------------------------------------
    async for e in _stage("fusion", "Fusing EEG + clinical text (cross-attention, MC-dropout)"):
        yield e
    text_emb = TextEncoder().encode(patient)
    # grounded priors bias the fusion heads toward the detector evidence
    sev_idx = {"minimal": 0, "mild": 1, "moderate": 2, "severe": 3, "critical": 4}.get(top.get("severity", "moderate"), 2)
    prior = {
        "severity_bias": np.eye(len(SEVERITY))[min(sev_idx + (1 if sz_prob > 0.7 else 0), 4)] * 2.0,
        "urgency_bias": np.eye(len(URGENCY))[URGENCY.index(urgency)] * 2.0,
        "conf_bias": (top.get("confidence", 0.6) - 0.5) * 4,
    }
    fusion = await asyncio.to_thread(MultimodalFusion().predict, eeg_emb, text_emb, prior)
    result["fusion"] = fusion
    result["severity"] = fusion["severity"]
    result["urgency"] = urgency
    result["confidence"] = fusion["confidence"]
    result["uncertainty"] = fusion["uncertainty"]
    result["models_used"]["fusion"] = "Cross-attention + MC-dropout"
    yield _ev(type="stage", agent="fusion", status="done",
              message=f"Severity {fusion['severity']} · confidence {fusion['confidence']:.2f} "
                      f"± {fusion['uncertainty']:.2f} (uncertainty) · attention→text {fusion['attention_to_text']:.2f}",
              data=fusion)

    # 8 ── Diagnostician -----------------------------------------------------
    async for e in _stage("diagnostician", "Reasoning over fused multimodal evidence"):
        yield e
    llm, cfg = get_llm("diagnostician", ctx.llm_override)
    dx_text = await llm.complete(_SYS["diagnostician"],
        f"top_diagnosis: {top.get('condition','unknown')}\nicd10: {top.get('icd10','')}\n"
        f"primary_finding: {det['primary_finding']}\nseizure_type: {det['seizure_type']}\n"
        f"eeg_markers: {', '.join(top.get('eeg_markers', []))}\nsymptoms: {symptoms}\n"
        f"seizure_probability: {sz_prob}\nsnn_event: {snn_out['detected_event']}\n"
        "Explain the leading diagnosis and key differentials in one paragraph.")
    result["primary_diagnosis"] = {
        "condition": top.get("condition", "Undetermined"), "icd10": top.get("icd10"),
        "confidence": fusion["confidence"], "reasoning": dx_text,
        "provider": cfg.provider, "model": cfg.model,
    }
    result["models_used"]["reasoning"] = f"{cfg.provider}:{cfg.model}"
    yield _ev(type="stage", agent="diagnostician", status="done", message=dx_text,
              data={"condition": top.get("condition"), "confidence": fusion["confidence"]})

    # 9 ── Pharmacologist ----------------------------------------------------
    async for e in _stage("pharmacologist", "Checking drug-drug interactions"):
        yield e
    di = await asyncio.to_thread(knowledge.drug_interactions, meds)
    result["drug_safety"] = di
    yield _ev(type="stage", agent="pharmacologist", status="done",
              message=(f"{di['count']} interaction(s); highest: {di['highest_severity']}" if di["count"]
                       else "No significant interactions detected"), data=di)

    # 10 ── Safety Critic ----------------------------------------------------
    async for e in _stage("critic", "Running guardrails, calibration & uncertainty check"):
        yield e
    flags = _safety_review(result)
    result["safety"] = flags
    yield _ev(type="stage", agent="critic", status="done",
              message="; ".join(flags["notes"]) or "All checks passed", data=flags)

    # 11 ── Reporter ---------------------------------------------------------
    async for e in _stage("reporter", "Synthesising the clinical report"):
        yield e
    llm, cfg = get_llm("reporter", ctx.llm_override)
    rep_text = await llm.complete(_SYS["reporter"],
        f"top_diagnosis: {top.get('condition','unknown')}\nseverity: {fusion['severity']}\n"
        f"urgency: {urgency}\nseizure_probability: {sz_prob}\nprimary_finding: {det['primary_finding']}\n"
        f"trend: {temporal['trend']}\nuncertainty: {fusion['uncertainty']}\n"
        "Write a concise clinical summary suitable for the chart.")
    result["summary"] = rep_text
    result["report_id"] = f"RPT-{abs(hash(str(result))) % 10**8:08d}"
    result["disclaimer"] = ("Decision-support only. Not FDA-approved. Requires review by a qualified clinician.")

    # backward-compatible `eeg` block for the existing charts
    result["eeg"] = _eeg_summary(feats, xf, fs, det)
    yield _ev(type="stage", agent="reporter", status="done", message=rep_text)

    yield _ev(type="result", data=result)
    yield _ev(type="done", message="Analysis complete")


def _eeg_summary(feats: dict, xf: np.ndarray, fs: float, det: dict) -> dict:
    rel = feats["spectral"]["rel_band_power"]
    seizure = next((d for d in det["detections"] if d["id"] == "seizure"), {})
    reasons = (seizure.get("explanation", "") or "").split("; ")
    return {
        "band_powers": rel,
        "dominant_band": max(rel, key=rel.get),
        "theta_beta_ratio": feats["spectral"]["band_ratios"]["theta_beta"],
        "spectral_entropy": feats["spectral"]["spectral_entropy"],
        "synchrony": feats["connectivity"]["mean_correlation"],
        "line_length": feats["time_domain"]["line_length"],
        "seizure_probability": det["seizure_probability"],
        "seizure_reasons": [r for r in reasons if r],
        "psd_preview": eeg._psd_preview(xf, fs),
        "waveform_preview": eeg._waveform_preview(xf, fs),
        "recording": {"channels": len(feats["channels"]), "sampling_rate": fs},
    }


async def _stage(agent: str, message: str) -> AsyncIterator[str]:
    yield _ev(type="stage", agent=agent, status="running", message=message)
    await asyncio.sleep(0.2)


def _safety_review(result: dict) -> dict:
    notes = []
    conf = result.get("confidence", 0)
    unc = result.get("uncertainty", 0)
    if conf < 0.6:
        notes.append("Low diagnostic confidence — recommend specialist confirmation")
    if unc > 0.12:
        notes.append(f"Elevated model uncertainty ({unc:.2f}) — interpret with caution")
    if result.get("drug_safety", {}).get("requires_action"):
        notes.append("Major/contraindicated drug interaction flagged — review before prescribing")
    if result.get("eeg", {}).get("seizure_probability", 0) >= 0.8:
        notes.append("High seizure probability — time-critical pathway")
    if result.get("primary_finding") == "burst_suppression":
        notes.append("Burst-suppression — assess sedation depth / anoxic injury urgently")
    if result.get("qeeg", {}).get("quality", {}).get("quality_score", 1) < 0.6:
        notes.append("Reduced recording quality — features may be unreliable")
    return {"passed": not notes, "notes": notes,
            "calibrated_confidence": round(min(conf, 0.95), 3)}


_SYS = {
    "triage": "You are a neuro-ICU triage clinician. Be decisive and concise. "
              "Always state an explicit urgency level (Routine/Urgent/Emergent).",
    "diagnostician": "You are a board-certified neurologist reasoning over EEG and clinical "
                     "data. Be precise, hedge appropriately, name differentials.",
    "reporter": "You are a clinical scribe. Produce a tight, structured chart summary. "
                "No fabricated values; rely only on the provided context.",
}


# ── chat copilot (unchanged contract) ───────────────────────────────────────
async def chat_stream(messages: list[dict], context: dict, llm_override: dict) -> AsyncIterator[str]:
    llm, cfg = get_llm("chat", llm_override)
    last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    lines = []
    if context.get("analysis"):
        a = context["analysis"]
        pd = a.get("primary_diagnosis", {})
        lines.append(f"current_diagnosis: {pd.get('condition')} ({pd.get('icd10')})")
        lines.append(f"primary_finding: {a.get('primary_finding')}")
        lines.append(f"seizure_probability: {a.get('eeg', {}).get('seizure_probability')}")
        lines.append(f"severity: {a.get('severity')} urgency: {a.get('urgency')}")
        if a.get("neuromorphic"):
            lines.append(f"snn_event: {a['neuromorphic'].get('detected_event')}")
    if context.get("patient"):
        lines.append(f"patient_symptoms: {context['patient'].get('symptoms')}")
    prompt = "\n".join(lines) + f"\nquestion: {last}"
    yield _ev(type="stage", agent="chat", status="running")
    async for tok in llm.stream(_CHAT_SYS, prompt):
        yield _ev(type="token", message=tok)
    yield _ev(type="done")


_CHAT_SYS = (
    "You are the Clinical AI Copilot assistant. You help clinicians interpret EEG analyses, "
    "diagnoses, drug-safety findings and the knowledge graph. Be accurate, cite the structured "
    "context, never invent patient data, and add a brief safety note when discussing treatment."
)
