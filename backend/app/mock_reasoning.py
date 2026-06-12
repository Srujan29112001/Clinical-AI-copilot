"""
Deterministic, key-free clinical reasoning for demo mode.

This is NOT a model — it composes clinically-plausible narrative text from the
structured context (EEG features + retrieved knowledge) that each agent is
handed. It exists so the entire multi-agent product is testable end-to-end and
deployable to the web with zero API keys, exactly the way Helix ships a working
public demo. Swap in a real provider per-request from the UI to get genuine
LLM reasoning.
"""
from __future__ import annotations

import re


def _grab(prompt: str, label: str, default: str = "") -> str:
    m = re.search(rf"{label}\s*:\s*(.+)", prompt, re.IGNORECASE)
    return m.group(1).strip() if m else default


def mock_response(role: str, system: str, prompt: str) -> str:
    role = role.lower()
    if role == "triage":
        return _triage(prompt)
    if role == "diagnostician":
        return _diagnostician(prompt)
    if role == "reporter":
        return _reporter(prompt)
    if role == "pharmacologist":
        return _pharmacologist(prompt)
    if role == "chat":
        return _chat(prompt)
    return (
        "[simulated] Based on the supplied EEG features and retrieved clinical "
        "evidence, the findings are consistent with the structured summary above. "
        "Configure a real LLM provider (local Ollama/vLLM or an API key) for "
        "full free-text reasoning."
    )


def _triage(prompt: str) -> str:
    sz = _grab(prompt, "seizure_probability", "0")
    try:
        p = float(sz)
    except ValueError:
        p = 0.0
    if p >= 0.8:
        return (
            "URGENCY: EMERGENT. The EEG shows a high probability of ictal activity. "
            "Escalate to the on-call neurologist immediately and prepare to administer "
            "a benzodiazepine per status-epilepticus protocol if clinical seizures are "
            "observed. Continuous monitoring is warranted."
        )
    if p >= 0.45:
        return (
            "URGENCY: URGENT. Epileptiform features are present at a level that warrants "
            "neurology review within the hour. Maintain monitoring and document any "
            "clinical correlates."
        )
    return (
        "URGENCY: ROUTINE. No acute ictal pattern dominates the recording. Proceed with "
        "standard interpretation and correlate with the clinical history."
    )


def _diagnostician(prompt: str) -> str:
    dx = _grab(prompt, "top_diagnosis", "a neurological condition")
    icd = _grab(prompt, "icd10", "")
    markers = _grab(prompt, "eeg_markers", "the observed spectral pattern")
    return (
        f"[simulated reasoning] The dominant EEG signature ({markers}) together with the "
        f"reported symptoms is most consistent with {dx}"
        + (f" (ICD-10 {icd})" if icd else "")
        + ". The band-power distribution and inter-channel synchrony reinforce this "
        "impression. Differential considerations should remain open pending clinical "
        "correlation, and uncertainty is reflected in the confidence score."
    )


def _pharmacologist(prompt: str) -> str:
    inter = _grab(prompt, "interactions", "none detected")
    meds = _grab(prompt, "medications", "the current regimen")
    if "none" in inter.lower():
        return (
            f"No significant drug–drug interactions were identified across {meds}. "
            "Standard therapeutic monitoring applies."
        )
    return (
        f"Reviewing {meds}: {inter}. Recommend the documented management steps and "
        "appropriate serum-level monitoring before any dose change."
    )


def _reporter(prompt: str) -> str:
    dx = _grab(prompt, "top_diagnosis", "the identified condition")
    sev = _grab(prompt, "severity", "moderate")
    urg = _grab(prompt, "urgency", "routine")
    return (
        "CLINICAL SUMMARY (simulated)\n"
        f"The multi-agent pipeline analysed the submitted recording and converged on "
        f"{dx} at {sev} severity with a {urg} disposition. EEG preprocessing removed "
        "artefacts and extracted spectral, connectivity and entropy features; the "
        "diagnostic model and the knowledge-graph retrieval agreed on the leading "
        "impression. Evidence-based treatment options and a drug-safety check are listed "
        "below. This output is decision-support only and requires clinician review."
    )


def _chat(prompt: str) -> str:
    q = _grab(prompt, "question", prompt).lower()
    if "seizure" in q or "epilep" in q:
        return (
            "[simulated assistant] Seizure likelihood in this recording is driven by the "
            "spike-wave activity and elevated high-frequency power flagged during signal "
            "analysis. If the probability is high, treat it as time-critical and involve "
            "neurology. Ask me to open the full report for the per-channel breakdown."
        )
    if "drug" in q or "medic" in q or "interaction" in q:
        return (
            "[simulated assistant] I checked the medication list against the antiepileptic "
            "interaction table. Any flagged pair, its mechanism and the recommended "
            "management are shown in the Pharmacology section of the report."
        )
    return (
        "[simulated assistant] I'm the Clinical AI Copilot. I can explain the EEG findings, "
        "the diagnosis reasoning, drug-safety checks, or the knowledge graph behind a result. "
        "Connect a local model (Ollama/vLLM) or an API key in Settings for full conversational "
        "reasoning."
    )
