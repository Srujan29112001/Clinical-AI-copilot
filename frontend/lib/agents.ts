import type { AgentDef } from "./types";

// Mirrors backend/app/agents.py AGENTS (used when running offline / for the UI).
export const AGENTS: AgentDef[] = [
  { id: "signal", name: "Signal Analyst", role: "EEG preprocessing & feature extraction", icon: "activity" },
  { id: "triage", name: "Triage Agent", role: "Acuity & urgency assessment", icon: "siren" },
  { id: "retriever", name: "Knowledge Retriever", role: "GraphRAG over ICD-10 / SNOMED / RxNorm", icon: "network" },
  { id: "diagnostician", name: "Diagnostician", role: "Differential diagnosis reasoning", icon: "stethoscope" },
  { id: "pharmacologist", name: "Pharmacologist", role: "Drug-drug interaction safety", icon: "pill" },
  { id: "critic", name: "Safety Critic", role: "Self-correction & guardrails", icon: "shield" },
  { id: "reporter", name: "Reporter", role: "Clinical report synthesis", icon: "file-text" },
];

export const AGENT_ORDER = AGENTS.map((a) => a.id);
