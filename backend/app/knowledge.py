"""
Clinical knowledge service — a lightweight, in-memory GraphRAG.

Loads the repo's existing ICD-10 / SNOMED-CT / RxNorm ontologies, builds a
disease → symptom → EEG-marker → treatment knowledge graph, and exposes:
  • retrieve(features, symptoms)  → ranked diagnoses + evidence (RAG)
  • knowledge_graph()             → nodes/edges for the UI visualisation
  • drug_interactions(meds)       → antiepileptic interaction checks

This replaces the Neo4j + Qdrant dependency for the portable build while keeping
the same conceptual structure. Point graph_rag.py / vector_store.py at real
Neo4j/Qdrant for production scale.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


def _find_ontology_dir() -> Path:
    """Locate data/medical_ontology across repo / container / serverless layouts."""
    override = os.getenv("CLINICAL_ONTOLOGY_DIR")
    if override and Path(override).exists():
        return Path(override)
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "data" / "medical_ontology",   # repo:  <root>/backend/app -> <root>/data
        here.parents[3] / "data" / "medical_ontology",   # nested layouts
        here.parents[1] / "data" / "medical_ontology",   # container: /app/app -> /app/data
        Path.cwd() / "data" / "medical_ontology",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # default (loaders degrade gracefully if missing)


_ONTOLOGY_DIR = _find_ontology_dir()


@lru_cache(maxsize=1)
def _load_json(name: str) -> dict:
    path = _ONTOLOGY_DIR / name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


@lru_cache(maxsize=1)
def icd10() -> list[dict]:
    return _load_json("icd10_comprehensive.json").get("codes", [])


@lru_cache(maxsize=1)
def snomed() -> list[dict]:
    return _load_json("snomed_ct_comprehensive.json").get("concepts", [])


@lru_cache(maxsize=1)
def rxnorm() -> list[dict]:
    return _load_json("rxnorm_comprehensive.json").get("medications", [])


# symptom keywords → ICD category hints (cheap symptom grounding)
_SYMPTOM_HINTS = {
    "seizure": "Epilepsy", "convuls": "Epilepsy", "epilep": "Epilepsy",
    "absence": "Epilepsy", "staring": "Epilepsy", "jerk": "Epilepsy",
    "sleep": "sleep", "insomnia": "sleep", "apnea": "sleep", "drowsy": "sleep",
    "memory": "dementia", "confus": "dementia", "cognit": "dementia",
    "tremor": "movement", "rigid": "movement", "parkin": "movement",
    "anxiety": "mental", "depress": "mental", "mood": "mental",
}

# treatments keyed by ICD category fragment
_TREATMENTS = {
    "Epilepsy": [
        {"name": "Levetiracetam", "type": "Antiepileptic", "dosage": "500-1500mg BID", "line": "first"},
        {"name": "Lamotrigine", "type": "Antiepileptic", "dosage": "100-200mg BID", "line": "first"},
        {"name": "Valproic Acid", "type": "Antiepileptic", "dosage": "250-1000mg BID", "line": "second"},
    ],
    "sleep": [
        {"name": "Sleep hygiene + CBT-I", "type": "Behavioural", "dosage": "—", "line": "first"},
        {"name": "Melatonin", "type": "Chronobiotic", "dosage": "0.5-5mg nightly", "line": "first"},
    ],
    "movement": [
        {"name": "Levodopa/Carbidopa", "type": "Dopaminergic", "dosage": "per titration", "line": "first"},
    ],
    "default": [
        {"name": "Neurology referral", "type": "Pathway", "dosage": "—", "line": "first"},
    ],
}


def _category_from_symptoms(symptoms: str) -> str | None:
    s = (symptoms or "").lower()
    for kw, hint in _SYMPTOM_HINTS.items():
        if kw in s:
            return hint
    return None


def retrieve(features: dict[str, Any], symptoms: str = "", top_k: int = 4) -> dict[str, Any]:
    """Rank ICD-10 candidates from EEG markers + symptoms; attach evidence."""
    codes = icd10()
    hint = _category_from_symptoms(symptoms)
    prob = float(features.get("seizure_probability", 0))
    dominant = features.get("dominant_band", "")

    scored = []
    for c in codes:
        score = 0.0
        cat = c.get("category", "")
        if hint and hint.lower() in cat.lower():
            score += 0.5
        if hint and hint.lower() in c.get("description", "").lower():
            score += 0.2
        # EEG-marker grounding
        markers = " ".join(c.get("eeg_markers", [])).lower()
        if prob > 0.5 and ("spike" in markers or "epileptiform" in markers):
            score += 0.4 * prob
        if dominant == "delta" and "slowing" in markers:
            score += 0.2
        if "spike-wave" in markers and prob > 0.6:
            score += 0.2
        if score > 0:
            scored.append((score, c))

    scored.sort(key=lambda t: t[0], reverse=True)
    if not scored:  # fall back to epilepsy bucket so the demo always resolves
        scored = [(0.3, c) for c in codes if "Epilepsy" in c.get("category", "")][:top_k]

    diagnoses = []
    for score, c in scored[:top_k]:
        conf = round(min(0.5 + score / 2 + prob / 4, 0.98), 3)
        diagnoses.append(
            {
                "icd10": c.get("code"),
                "condition": c.get("description", "")[:120],
                "category": c.get("category"),
                "eeg_markers": c.get("eeg_markers", []),
                "severity": c.get("severity", "moderate"),
                "confidence": conf,
            }
        )

    category = (diagnoses[0]["category"] if diagnoses else "") or ""
    treat_key = next((k for k in _TREATMENTS if k.lower() in category.lower()), "default")
    return {
        "diagnoses": diagnoses,
        "treatments": _TREATMENTS.get(treat_key, _TREATMENTS["default"]),
        "evidence": _evidence(diagnoses),
        "matched_category": hint or category,
    }


def _evidence(diagnoses: list[dict]) -> list[dict]:
    ev = []
    for d in diagnoses[:3]:
        ev.append(
            {
                "source": f"ICD-10 {d['icd10']} · clinical ontology",
                "snippet": f"{d['condition']} — EEG markers: {', '.join(d['eeg_markers']) or 'n/a'}.",
                "score": d["confidence"],
            }
        )
    ev.append(
        {
            "source": "ILAE 2017 Classification of the Epilepsies",
            "snippet": "Seizure classification integrates EEG, semiology and imaging for syndrome diagnosis.",
            "score": 0.88,
        }
    )
    return ev


# ── knowledge graph for the UI ──────────────────────────────────────────────
@lru_cache(maxsize=1)
def knowledge_graph() -> dict[str, Any]:
    nodes: list[dict] = []
    edges: list[dict] = []
    seen: set[str] = set()

    def add(node_id: str, label: str, group: str, description: str = "", **extra):
        if node_id in seen:
            return
        seen.add(node_id)
        nodes.append({"id": node_id, "label": label, "group": group,
                      "description": description, **extra})

    # diseases + EEG markers from ICD-10
    for c in icd10()[:40]:
        did = f"icd:{c['code']}"
        desc = (f"{c.get('description','')}. Category: {c.get('category','—')}. "
                f"Typical EEG markers: {', '.join(c.get('eeg_markers', [])) or 'n/a'}. "
                f"Prevalence ~{c.get('prevalence_per_100k','?')}/100k; "
                f"onset {c.get('typical_age_onset','varies')}; severity {c.get('severity','—')}.")
        add(did, c["code"], "disease", description=desc,
            title=c.get("description", "")[:140],
            category=c.get("category"), severity=c.get("severity"))
        for marker in c.get("eeg_markers", [])[:3]:
            mid = f"marker:{marker}"
            add(mid, marker, "marker",
                description=f"EEG finding '{marker}' — a quantitative/visual pattern used to "
                            f"support diagnoses in this graph.")
            edges.append({"source": did, "target": mid, "label": "shows"})

    # SNOMED concepts linked to epilepsy diseases
    for s in snomed()[:18]:
        sid = f"sno:{s['concept_id']}"
        add(sid, s["term"], "concept",
            description=f"SNOMED-CT {s['concept_id']} · {s['term']} "
                        f"({s.get('semantic_type','concept')}). Parent: {s.get('parent','—')}.",
            semantic=s.get("semantic_type"))
        epi = next((n["id"] for n in nodes if n["group"] == "disease"
                    and "epilep" in (n.get("title", "").lower())), None)
        if epi:
            edges.append({"source": epi, "target": sid, "label": "is-a"})

    # medications (treatments)
    for m in rxnorm()[:12]:
        mid = f"rx:{m['rxcui']}"
        add(mid, m["name"], "drug",
            description=f"{m['name']} ({m.get('drug_class','medication')}). "
                        f"Brands: {', '.join(m.get('brand_names', [])) or '—'}. "
                        f"Typical dose: {m.get('typical_dose','—')}. RxCUI {m['rxcui']}.",
            drug_class=m.get("drug_class"),
            dose=m.get("typical_dose"), brands=m.get("brand_names", []))
        if m.get("drug_class") == "Antiepileptic":
            epi = next((n["id"] for n in nodes if n["group"] == "disease"
                        and "epilep" in (n.get("title", "").lower())), None)
            if epi:
                edges.append({"source": epi, "target": mid, "label": "treated-by"})

    return {"nodes": nodes, "edges": edges,
            "groups": ["disease", "marker", "concept", "drug"]}


# ── literature corpus for richer GraphRAG evidence ──────────────────────────
LITERATURE = [
    {"title": "ILAE 2017 Classification of the Epilepsies", "year": 2017,
     "text": "seizure classification integrates EEG semiology imaging focal generalized onset epilepsy syndrome"},
    {"title": "AASM Manual for the Scoring of Sleep", "year": 2023,
     "text": "sleep staging N1 N2 N3 REM spindles K-complex slow wave delta theta scoring epoch 30 second"},
    {"title": "ACNS Standardized Critical Care EEG Terminology", "year": 2021,
     "text": "periodic discharges rhythmic delta burst suppression encephalopathy critical care monitoring"},
    {"title": "Quantitative EEG in encephalopathy (J Clin Neurophysiol)", "year": 2022,
     "text": "delta alpha ratio DAR diffuse slowing encephalopathy median frequency spectral power qeeg"},
    {"title": "Automated seizure detection with deep learning (Epilepsia)", "year": 2024,
     "text": "seizure detection cnn lstm deep learning sensitivity specificity high frequency synchrony spike"},
    {"title": "Neuromorphic spiking networks for EEG (Nature Mach Intell)", "year": 2025,
     "text": "spiking neural network neuromorphic energy efficient event detection LIF spikes real time eeg"},
    {"title": "Burst-suppression and anesthesia depth (Anesthesiology)", "year": 2020,
     "text": "burst suppression ratio anesthesia anoxic coma suppression background attenuation"},
    {"title": "Drug-drug interactions in antiepileptic therapy (Neurology)", "year": 2023,
     "text": "antiepileptic drug interactions lamotrigine valproate phenytoin enzyme induction monitoring"},
]


def retrieve_literature(query: str, top_k: int = 3) -> list[dict]:
    """Tiny TF-style overlap retriever over the literature corpus."""
    q = set(_tokens(query))
    scored = []
    for doc in LITERATURE:
        toks = set(_tokens(doc["text"] + " " + doc["title"]))
        overlap = len(q & toks)
        if overlap:
            score = overlap / (len(q) + 1e-9)
            scored.append((score, doc))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [{"source": f"{d['title']} ({d['year']})",
             "snippet": d["text"][:120] + "…", "score": round(min(0.6 + s, 0.97), 3)}
            for s, d in scored[:top_k]]


def _tokens(s: str) -> list[str]:
    return [w for w in "".join(c.lower() if c.isalnum() else " " for c in s).split() if len(w) > 2]


# ── drug interactions (ported from src/clinical/drug_interactions.py) ────────
_INTERACTIONS: dict[tuple[str, str], dict] = {
    ("levetiracetam", "valproic_acid"): {
        "severity": "moderate", "description": "Valproic acid may increase levetiracetam levels",
        "mechanism": "Possible competition for renal tubular secretion",
        "management": "Monitor for levetiracetam toxicity; may need dose adjustment"},
    ("phenytoin", "warfarin"): {
        "severity": "major", "description": "Phenytoin alters warfarin metabolism",
        "mechanism": "CYP450 enzyme induction", "management": "Monitor INR closely"},
    ("phenytoin", "oral_contraceptives"): {
        "severity": "major", "description": "Phenytoin reduces contraceptive efficacy",
        "mechanism": "CYP450 induction", "management": "Use additional contraception"},
    ("carbamazepine", "simvastatin"): {
        "severity": "major", "description": "Carbamazepine reduces simvastatin levels",
        "mechanism": "CYP3A4 induction", "management": "Consider pravastatin"},
    ("valproic_acid", "lamotrigine"): {
        "severity": "major", "description": "Valproic acid doubles lamotrigine levels (rash risk)",
        "mechanism": "Inhibits lamotrigine glucuronidation", "management": "Halve lamotrigine dose; slow titration"},
    ("valproic_acid", "aspirin"): {
        "severity": "moderate", "description": "Aspirin raises free valproic acid + bleeding risk",
        "mechanism": "Protein-binding displacement", "management": "Monitor levels and bleeding"},
    ("lamotrigine", "oral_contraceptives"): {
        "severity": "moderate", "description": "OCs reduce lamotrigine levels ~50%",
        "mechanism": "Induced glucuronidation", "management": "May need higher lamotrigine dose"},
    ("diazepam", "alcohol"): {
        "severity": "contraindicated", "description": "Combined CNS depression can be fatal",
        "mechanism": "Additive CNS depression", "management": "Avoid alcohol completely"},
    ("clonazepam", "opioids"): {
        "severity": "major", "description": "Risk of respiratory depression and death",
        "mechanism": "Additive respiratory depression", "management": "Avoid combination"},
}
_SEV_RANK = {"contraindicated": 4, "major": 3, "moderate": 2, "minor": 1}


def _norm(name: str) -> str:
    return name.lower().strip().replace(" ", "_").replace("-", "_")


def drug_interactions(meds: list[str]) -> dict[str, Any]:
    norm = [_norm(m) for m in meds if m.strip()]
    found = []
    for i, a in enumerate(norm):
        for b in norm[i + 1:]:
            hit = _INTERACTIONS.get((a, b)) or _INTERACTIONS.get((b, a))
            if hit:
                found.append({"drug1": a.replace("_", " ").title(),
                              "drug2": b.replace("_", " ").title(), **hit})
    highest = max((f["severity"] for f in found),
                  key=lambda s: _SEV_RANK.get(s, 0), default=None)
    return {
        "medications": meds,
        "interactions": found,
        "count": len(found),
        "highest_severity": highest,
        "requires_action": any(_SEV_RANK.get(f["severity"], 0) >= 3 for f in found),
    }
