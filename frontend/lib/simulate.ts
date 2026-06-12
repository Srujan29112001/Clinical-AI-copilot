/**
 * Client-side simulation of the multi-agent pipeline.
 *
 * Mirrors backend/app/agents.py so the deployed web app is fully testable with
 * NO backend (the Helix "Studio simulates locally" pattern). When a backend is
 * configured via NEXT_PUBLIC_API_URL, the real SSE pipeline is used instead.
 */
import { AGENT_ORDER } from "./agents";
import type { AgentEvent, AnalysisResult, BandPowers, PatientContext } from "./types";
import { sleep } from "./utils";

type Profile = "ictal" | "normal" | "sleep" | "upload";

const ICD = {
  ictal: { code: "G40.3", name: "Generalized idiopathic epilepsy and epileptic syndromes", sev: "moderate", markers: ["generalized spike-wave", "3 Hz spike-wave"] },
  focal: { code: "G40.0", name: "Localization-related (focal) idiopathic epilepsy with seizures of localized onset", sev: "moderate", markers: ["focal spikes", "sharp waves"] },
  sleep: { code: "G47.0", name: "Insomnia / disorders of initiating and maintaining sleep", sev: "mild", markers: ["theta slowing", "reduced spindles"] },
  normal: { code: "R94.01", name: "Abnormal electroencephalogram — none; study within normal limits", sev: "minimal", markers: ["posterior dominant alpha"] },
};

const INTERACTIONS: Record<string, { severity: string; description: string; mechanism: string; management: string }> = {
  "levetiracetam|valproic acid": { severity: "moderate", description: "Valproic acid may increase levetiracetam levels", mechanism: "Renal tubular competition", management: "Monitor for levetiracetam toxicity" },
  "phenytoin|warfarin": { severity: "major", description: "Phenytoin alters warfarin metabolism", mechanism: "CYP450 induction", management: "Monitor INR closely" },
  "valproic acid|lamotrigine": { severity: "major", description: "Valproic acid doubles lamotrigine levels (rash risk)", mechanism: "Inhibits glucuronidation", management: "Halve lamotrigine dose; slow titration" },
  "lamotrigine|phenytoin": { severity: "moderate", description: "Phenytoin lowers lamotrigine levels", mechanism: "CYP induction", management: "May need higher lamotrigine dose" },
};

function seededRand(seed: number) {
  let s = seed % 2147483647;
  if (s <= 0) s += 2147483646;
  return () => (s = (s * 16807) % 2147483647) / 2147483647;
}

function bands(profile: Profile, rnd: () => number): BandPowers {
  const j = () => (rnd() - 0.5) * 0.04;
  if (profile === "ictal") return norm({ delta: 0.08 + j(), theta: 0.04 + j(), alpha: 0.05 + j(), beta: 0.55 + j(), gamma: 0.23 + j() });
  if (profile === "sleep") return norm({ delta: 0.18 + j(), theta: 0.55 + j(), alpha: 0.12 + j(), beta: 0.1 + j(), gamma: 0.05 + j() });
  // normal or generic upload
  return norm({ delta: 0.12 + j(), theta: 0.16 + j(), alpha: 0.45 + j(), beta: 0.19 + j(), gamma: 0.08 + j() });
}
function norm(b: BandPowers): BandPowers {
  const t = b.delta + b.theta + b.alpha + b.beta + b.gamma;
  return { delta: r(b.delta / t), theta: r(b.theta / t), alpha: r(b.alpha / t), beta: r(b.beta / t), gamma: r(b.gamma / t) };
}
const r = (x: number) => Math.round(x * 1e4) / 1e4;

function seizureProb(b: BandPowers, sync: number): { p: number; reasons: string[] } {
  const reasons: string[] = [];
  let s = 0;
  const hf = b.beta + b.gamma;
  if (hf > 0.5) { s += 0.45; reasons.push(`recruiting high-frequency power (${Math.round(hf * 100)}%)`); }
  else if (hf > 0.3) { s += 0.22; reasons.push(`elevated high-frequency power (${Math.round(hf * 100)}%)`); }
  if (sync > 0.8) { s += 0.35; reasons.push(`hyper-synchrony across channels (${sync.toFixed(2)})`); }
  else if (sync > 0.6) { s += 0.12; reasons.push(`raised inter-channel synchrony (${sync.toFixed(2)})`); }
  if (hf > 0.4) { s += 0.1; reasons.push("rhythmic high-frequency discharge"); }
  if (b.delta > 0.5) { s += 0.08; reasons.push(`excess slow-wave power (${Math.round(b.delta * 100)}%)`); }
  if (reasons.length === 0) reasons.push("spectral profile within normal limits");
  return { p: r(Math.min(s, 0.99)), reasons };
}

function psd(b: BandPowers, rnd: () => number) {
  const out: Array<{ freq: number; power: number }> = [];
  const centers: Array<[number, number]> = [[2, b.delta], [6, b.theta], [10, b.alpha], [20, b.beta], [38, b.gamma]];
  for (let f = 0.5; f <= 50; f += 0.8) {
    let p = 0;
    for (const [c, amp] of centers) p += amp * Math.exp(-((f - c) ** 2) / (2 * 9));
    p += 0.02 * rnd();
    out.push({ freq: r(f), power: r(p) });
  }
  const max = Math.max(...out.map((o) => o.power)) || 1;
  return out.map((o) => ({ freq: o.freq, power: r(o.power / max) }));
}

function waveform(profile: Profile, rnd: () => number) {
  const out: Array<{ t: number; v: number }> = [];
  const N = 240, secs = 4;
  for (let i = 0; i < N; i++) {
    const t = (i / N) * secs;
    let v: number;
    if (profile === "ictal") {
      v = 0.5 * Math.sin(2 * Math.PI * 3 * t) + (((t * 3) % 1) < 0.05 ? 1 : 0) + 0.5 * Math.sin(2 * Math.PI * 24 * t);
    } else if (profile === "sleep") {
      v = 0.7 * Math.sin(2 * Math.PI * 6 * t) + 0.3 * Math.sin(2 * Math.PI * 2 * t);
    } else {
      v = 0.7 * Math.sin(2 * Math.PI * 10 * t) + 0.2 * Math.sin(2 * Math.PI * 18 * t);
    }
    v += (rnd() - 0.5) * 0.25;
    out.push({ t: r(t), v: r(v) });
  }
  const max = Math.max(...out.map((o) => Math.abs(o.v))) || 1;
  return out.map((o) => ({ t: o.t, v: r(o.v / max) }));
}

function checkInteractions(meds: string[]) {
  const found: AnalysisResult["drug_safety"]["interactions"] = [];
  const m = meds.map((x) => x.toLowerCase().trim()).filter(Boolean);
  for (let i = 0; i < m.length; i++)
    for (let k = i + 1; k < m.length; k++) {
      const hit = INTERACTIONS[`${m[i]}|${m[k]}`] || INTERACTIONS[`${m[k]}|${m[i]}`];
      if (hit) found.push({ drug1: cap(m[i]), drug2: cap(m[k]), ...hit });
    }
  const rank: Record<string, number> = { contraindicated: 4, major: 3, moderate: 2, minor: 1 };
  const highest = found.length ? found.reduce((a, b) => (rank[b.severity] > rank[a.severity] ? b : a)).severity : null;
  return { medications: meds, interactions: found, count: found.length, highest_severity: highest, requires_action: found.some((f) => rank[f.severity] >= 3) };
}
const cap = (s: string) => s.replace(/\b\w/g, (c) => c.toUpperCase());

export function buildResult(profile: Profile, patient: PatientContext, recording?: Record<string, unknown>): AnalysisResult {
  const rnd = seededRand(profile.length * 97 + patient.symptoms.length * 13 + 7);
  const b = bands(profile, rnd);
  const sync = profile === "ictal" ? 0.95 + rnd() * 0.04 : 0.45 + rnd() * 0.15;
  const { p, reasons } = seizureProb(b, sync);
  const symptomsLc = patient.symptoms.toLowerCase();
  const dxKey = p >= 0.8 ? "ictal" : p >= 0.45 ? "focal" : symptomsLc.includes("sleep") || profile === "sleep" ? "sleep" : "normal";
  const dx = ICD[dxKey as keyof typeof ICD];
  const urgency = p >= 0.8 ? "Emergent" : p >= 0.45 ? "Urgent" : "Routine";
  const confidence = r(Math.min(0.55 + p / 3 + rnd() * 0.1, 0.97));
  const drug = checkInteractions(patient.medications || []);

  const dominant = (Object.entries(b).sort((a, c) => c[1] - a[1])[0][0]) as string;
  const entropy = r(profile === "ictal" ? 0.4 + rnd() * 0.08 : 0.7 + rnd() * 0.15);

  const treatments = dxKey === "sleep"
    ? [{ name: "Sleep hygiene + CBT-I", type: "Behavioural", dosage: "—", line: "first" }, { name: "Melatonin", type: "Chronobiotic", dosage: "0.5-5mg nightly", line: "first" }]
    : dxKey === "normal"
      ? [{ name: "No pharmacotherapy indicated", type: "Pathway", dosage: "—", line: "first" }, { name: "Routine follow-up", type: "Pathway", dosage: "—", line: "first" }]
      : [{ name: "Levetiracetam", type: "Antiepileptic", dosage: "500-1500mg BID", line: "first" }, { name: "Lamotrigine", type: "Antiepileptic", dosage: "100-200mg BID", line: "first" }, { name: "Valproic Acid", type: "Antiepileptic", dosage: "250-1000mg BID", line: "second" }];

  const notes: string[] = [];
  if (confidence < 0.6) notes.push("Low diagnostic confidence — recommend specialist confirmation");
  if (drug.requires_action) notes.push("Major/contraindicated drug interaction flagged — review before prescribing");
  if (p >= 0.8) notes.push("High seizure probability — time-critical pathway");

  return {
    report_id: `RPT-${Math.floor(rnd() * 1e8).toString().padStart(8, "0")}`,
    patient_id: patient.patient_id,
    urgency, severity: dx.sev,
    summary: `The multi-agent pipeline analysed the recording and converged on ${dx.name} at ${dx.sev} severity with a ${urgency.toLowerCase()} disposition. EEG preprocessing removed artefacts and extracted spectral, connectivity and entropy features; the diagnostic reasoning and knowledge-graph retrieval agreed on the leading impression. This output is decision-support only and requires clinician review.`,
    disclaimer: "Decision-support only. Not FDA-approved. Requires review by a qualified clinician.",
    primary_diagnosis: {
      condition: dx.name, icd10: dx.code, confidence,
      reasoning: `The dominant EEG signature (${dx.markers.join(", ")}) together with the reported symptoms is most consistent with ${dx.name}. Band-power distribution (${dominant}-dominant) and inter-channel synchrony (${sync.toFixed(2)}) reinforce this impression; differentials remain open pending clinical correlation.`,
      provider: "simulated",
    },
    diagnoses: [
      { icd10: dx.code, condition: dx.name, category: dxKey === "sleep" ? "Sleep disorders" : "Epilepsy and recurrent seizures", eeg_markers: dx.markers, severity: dx.sev, confidence },
      { icd10: "G40.2", condition: "Localization-related symptomatic epilepsy with complex partial seizures", category: "Epilepsy and recurrent seizures", eeg_markers: ["temporal lobe spikes"], severity: "moderate", confidence: r(confidence - 0.18) },
    ],
    treatments,
    evidence: [
      { source: `ICD-10 ${dx.code} · clinical ontology`, snippet: `${dx.name} — EEG markers: ${dx.markers.join(", ")}.`, score: confidence },
      { source: "ILAE 2017 Classification of the Epilepsies", snippet: "Seizure classification integrates EEG, semiology and imaging for syndrome diagnosis.", score: 0.88 },
      { source: "Journal of Clinical Neurophysiology", snippet: "Quantitative EEG band-power and synchrony features support automated seizure screening.", score: 0.83 },
    ],
    triage: { urgency, narrative: triageText(urgency, p), provider: "simulated" },
    eeg: {
      band_powers: b, dominant_band: dominant,
      theta_beta_ratio: r(b.theta / (b.beta + 1e-9)),
      spectral_entropy: entropy, synchrony: r(sync),
      line_length: r(20 + p * 60 + rnd() * 10),
      seizure_probability: p, seizure_reasons: reasons,
      psd_preview: psd(b, rnd), waveform_preview: waveform(profile, rnd),
      recording: recording || { source: `sample:${profile}`, channels: 16, samples: 2048, duration_s: 8 },
    },
    drug_safety: drug,
    safety: { passed: notes.length === 0, notes, calibrated_confidence: r(Math.min(confidence, 0.95)) },
  };
}

function triageText(urgency: string, p: number) {
  if (urgency === "Emergent") return `URGENCY: EMERGENT. The EEG shows a high probability of ictal activity (p=${p.toFixed(2)}). Escalate to the on-call neurologist immediately and follow status-epilepticus protocol if clinical seizures are observed.`;
  if (urgency === "Urgent") return `URGENCY: URGENT. Epileptiform features are present (p=${p.toFixed(2)}) warranting neurology review within the hour. Maintain monitoring.`;
  return `URGENCY: ROUTINE. No acute ictal pattern dominates the recording (p=${p.toFixed(2)}). Proceed with standard interpretation and clinical correlation.`;
}

/** Stream simulated agent events with the same shape as the backend SSE. */
export async function* simulateStream(
  profile: Profile,
  patient: PatientContext,
  recording?: Record<string, unknown>,
): AsyncGenerator<AgentEvent> {
  const result = buildResult(profile, patient, recording);
  yield { type: "log", message: "Pipeline initialised — 7 agents online (offline demo mode)" };
  const messages: Record<string, string> = {
    signal: `Extracted 5 bands · seizure p=${result.eeg.seizure_probability.toFixed(2)}`,
    triage: result.triage.narrative,
    retriever: `${result.diagnoses.length} candidate diagnoses · top: ${result.primary_diagnosis.condition.slice(0, 50)}`,
    diagnostician: result.primary_diagnosis.reasoning,
    pharmacologist: result.drug_safety.count ? `${result.drug_safety.count} interaction(s); highest: ${result.drug_safety.highest_severity}` : "No significant interactions detected",
    critic: result.safety.notes.join("; ") || "All checks passed",
    reporter: result.summary,
  };
  for (const id of AGENT_ORDER) {
    yield { type: "stage", agent: id, status: "running", message: runningMsg(id) };
    await sleep(420 + Math.random() * 380);
    yield { type: "stage", agent: id, status: "done", message: messages[id] };
  }
  yield { type: "result", data: result as unknown as Record<string, unknown> };
  yield { type: "done", message: "Analysis complete" };
}

function runningMsg(id: string): string {
  return ({
    signal: "Filtering, artefact handling & feature extraction",
    triage: "Assessing clinical urgency",
    retriever: "Retrieving evidence from the knowledge graph",
    diagnostician: "Reasoning over fused EEG + clinical evidence",
    pharmacologist: "Checking drug-drug interactions",
    critic: "Running guardrails & calibration check",
    reporter: "Synthesising the clinical report",
  } as Record<string, string>)[id] || "Working…";
}
