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

  // ── v3 extras (mirror the backend so the offline demo shows everything) ──
  const v3 = buildV3(profile, b, sync, p, entropy, dxKey, confidence, rnd, symptomsLc);

  const notes: string[] = [];
  if (confidence < 0.6) notes.push("Low diagnostic confidence — recommend specialist confirmation");
  if (v3.fusion.uncertainty > 0.12) notes.push(`Elevated model uncertainty (${v3.fusion.uncertainty.toFixed(2)}) — interpret with caution`);
  if (drug.requires_action) notes.push("Major/contraindicated drug interaction flagged — review before prescribing");
  if (p >= 0.8) notes.push("High seizure probability — time-critical pathway");
  if (v3.primary_finding === "burst_suppression") notes.push("Burst-suppression — assess sedation depth / anoxic injury urgently");

  return {
    ...v3,
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

const EVENT_CLASSES = [
  "spike-wave complex", "sharp transient", "rhythmic discharge", "slow-wave burst",
  "muscle artifact", "background rhythm", "attenuation/suppression", "K-complex/spindle",
];
const CHANS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "P3", "Pz", "P4", "O1"];

function buildV3(profile: string, b: BandPowers, sync: number, p: number, entropy: number,
                 dxKey: string, confidence: number, rnd: () => number, symptomsLc: string) {
  const hf = b.beta + b.gamma;
  const dar = b.delta / (b.alpha + 1e-9);
  const plv = profile === "ictal" ? 0.75 + rnd() * 0.2 : 0.3 + rnd() * 0.3;
  const spikeRate = profile === "ictal" ? 0.8 + rnd() : rnd() * 0.3;
  const suppression = profile === "burst" ? 0.4 + rnd() * 0.3 : rnd() * 0.1;
  const lateralization = profile === "ictal" && rnd() > 0.5 ? "left" : "symmetric";

  // event-evidence → SNN raster (8 neurons × 30 steps)
  const evidence = [
    Math.min(p * 0.9 + spikeRate * 0.2, 0.98), Math.min(spikeRate * 0.6 + 0.1, 0.95),
    Math.min(plv * 0.7 + hf * 0.4, 0.97), Math.min(b.delta * 1.3, 0.95),
    Math.min((b.gamma) * 1.5, 0.9), Math.min(b.alpha * 1.4, 0.9),
    Math.min(suppression * 1.6, 0.95), Math.min((profile === "sleep" ? 0.6 : 0.1) + b.theta * 0.4, 0.95),
  ];
  const raster: number[][] = evidence.map((e) => Array.from({ length: 30 }, () => (rnd() < e ? 1 : 0) as number));
  const counts = raster.map((row) => row.reduce((a, c) => a + c, 0));
  const detectedIdx = counts.indexOf(Math.max(...counts));
  const totalSpikes = counts.reduce((a, c) => a + c, 0) + Math.round(40 + rnd() * 40);

  const trendSlope = profile === "ictal" ? 0.08 + rnd() * 0.06 : (rnd() - 0.5) * 0.08;
  const trend = trendSlope > 0.05 ? "escalating" : trendSlope < -0.05 ? "resolving" : "stable";
  const stateSeries = Array.from({ length: 8 }, (_, i) => r(1 + i * trendSlope + rnd() * 0.3));

  const sevIdx = ["minimal", "mild", "moderate", "severe", "critical"].indexOf(
    dxKey === "ictal" ? (p >= 0.8 ? "severe" : "moderate") : dxKey === "sleep" || dxKey === "normal" ? "mild" : "moderate");
  const sevDist = softDist(5, Math.max(sevIdx, 0), rnd);
  const urgIdx = p >= 0.8 ? 2 : p >= 0.45 ? 1 : 0;
  const urgDist = softDist(3, urgIdx, rnd);
  const uncertainty = r(0.02 + (1 - confidence) * 0.15 + rnd() * 0.03);

  const detections = [
    mkDet("seizure", "Ictal seizure activity", p, p >= 0.45, p >= 0.8 ? "critical" : "moderate",
      { type: p >= 0.8 ? "generalized" : p >= 0.45 ? "focal" : "none", lateralization },
      p >= 0.45 ? `high-frequency power ${pctNum(hf)}, PLV ${plv.toFixed(2)}` : "no ictal pattern"),
    mkDet("sleep", `Sleep stage: ${profile === "sleep" ? "N2" : "Wake"}`, profile === "sleep" ? 0.75 : 0.5,
      profile === "sleep", "minimal", { stage: profile === "sleep" ? "N2" : "Wake" },
      profile === "sleep" ? "theta background + spindles" : "alpha-dominant, awake"),
    mkDet("encephalopathy", "Diffuse slowing / encephalopathy", r(Math.min(Math.max((dar - 1.5) / 4.5, 0) * (profile === "sleep" ? 0.3 : 1), 0.99)),
      dar > 3 && profile !== "sleep", "moderate", { DAR: r(dar) }, `delta/alpha ratio ${dar.toFixed(1)}`),
    mkDet("burst_suppression", "Burst-suppression", r(Math.min(suppression * 1.5, 0.99)), profile === "burst",
      "critical", { suppression_ratio: r(suppression) }, profile === "burst" ? `suppression ratio ${pctNum(suppression)}` : "continuous background"),
    mkDet("focal", "Focal abnormality / lateralization", lateralization !== "symmetric" ? 0.6 : 0.2,
      lateralization !== "symmetric", "moderate", { lateralization }, lateralization !== "symmetric" ? `${lateralization} lateralization` : "symmetric"),
    mkDet("pdr", "Posterior dominant rhythm", profile === "normal" ? 0.9 : 0.2, profile === "normal", "minimal",
      { frequency_hz: 10 }, profile === "normal" ? "PDR present at 10 Hz (normal variant)" : "no organized posterior alpha"),
    mkDet("quality", "Recording quality", 0.85, false, "minimal", { quality_score: 0.85 }, "good signal quality"),
  ];
  const abnormal = detections.filter((d) => !["pdr", "quality", "sleep"].includes(d.id) && d.present);
  const primary_finding = abnormal.length ? abnormal.sort((a, c) => c.score - a.score)[0].id
    : profile === "sleep" ? "sleep" : "normal";

  const qeeg = {
    n_features: 50, channels: CHANS,
    spectral: {
      rel_band_power: b,
      abs_band_power: { delta: r(b.delta * 12), theta: r(b.theta * 12), alpha: r(b.alpha * 12), beta: r(b.beta * 12), gamma: r(b.gamma * 12) },
      band_ratios: { theta_beta: r(b.theta / (b.beta + 1e-9)), delta_alpha: r(dar), delta_theta: r(b.delta / (b.theta + 1e-9)), alpha_delta: r(b.alpha / (b.delta + 1e-9)) },
      sef95: r(profile === "ictal" ? 32 + rnd() * 8 : 18 + rnd() * 6), median_freq: r(profile === "encephalopathy" ? 3 + rnd() * 2 : 8 + rnd() * 3),
      peak_freq: r(profile === "ictal" ? 24 : profile === "sleep" ? 6 : 10), spectral_entropy: entropy,
    },
    time_domain: { rms: r(20 + rnd() * 30), line_length: r(20 + p * 60), zero_crossing_rate: r(50 + rnd() * 100),
      kurtosis: r((rnd() - 0.5) * 2), skewness: r((rnd() - 0.5)), activity: r(rnd()), mobility: r(0.2 + rnd() * 0.3), complexity: r(1 + rnd()) },
    complexity: { shannon_entropy: r(0.6 + rnd() * 0.3), permutation_entropy: r(0.7 + rnd() * 0.2), sample_entropy: r(0.8 + rnd()), approximate_entropy: r(0.5 + rnd() * 0.5) },
    connectivity: { mean_correlation: r(sync), alpha_coherence: r(0.3 + rnd() * 0.4), plv: r(plv), pli: r(0.2 + rnd() * 0.3) },
    asymmetry: { index: r(lateralization === "left" ? 0.2 : (rnd() - 0.5) * 0.1), lateralization, per_band: {} },
    events: { spike_count: Math.round(spikeRate * 10), spike_rate_hz: r(spikeRate) },
    states: { suppression_ratio: r(suppression), burst_suppression: profile === "burst" },
    quality: { flatline_frac: 0, clipping_frac: 0, emg_index: r(b.gamma), blink_index: r(b.delta * 0.5), quality_score: 0.85 },
    topography: CHANS.map((ch, i) => {
      const jit = () => (rnd() - 0.5) * 0.06;
      const tb = norm({ delta: b.delta + jit(), theta: b.theta + jit(), alpha: b.alpha + jit() + (i > 11 ? 0.1 : 0), beta: b.beta + jit(), gamma: b.gamma + jit() });
      return { channel: ch, rel_band_power: tb, dominant: (Object.entries(tb).sort((a, c) => c[1] - a[1])[0][0]) };
    }),
  };

  return {
    confidence, uncertainty, primary_finding,
    models_used: { signal: "CNN-LSTM encoder", events: "Spiking NN (LIF)", temporal: "Mamba2 SSM",
      knowledge: "GraphRAG + literature", fusion: "Cross-attention + MC-dropout", reasoning: "simulated (offline)" },
    qeeg, detections,
    neuromorphic: {
      top_events: [...counts].map((c, i) => ({ event: EVENT_CLASSES[i], activation: r(c / (totalSpikes + 1e-9)), spikes: c }))
        .sort((a, c) => c.spikes - a.spikes).slice(0, 4),
      detected_event: EVENT_CLASSES[detectedIdx], firing_rate: r(0.15 + rnd() * 0.15),
      total_spikes: totalSpikes, energy_uj: r(totalSpikes * 0.9e-3),
      raster, raster_labels: EVENT_CLASSES, steps: 30, neurons: 8,
    },
    temporal: { state_norm_series: stateSeries, trend_slope: r(trendSlope), trend, windows: 8 },
    fusion: {
      severity: ["minimal", "mild", "moderate", "severe", "critical"][argmax(sevDist)],
      severity_dist: distObj(["minimal", "mild", "moderate", "severe", "critical"], sevDist),
      urgency: ["Routine", "Urgent", "Emergent"][argmax(urgDist)],
      urgency_dist: distObj(["Routine", "Urgent", "Emergent"], urgDist),
      confidence, uncertainty, attention_to_text: r(0.3 + rnd() * 0.3), mc_samples: 30,
    },
  };
}

function mkDet(id: string, label: string, score: number, present: boolean, severity_hint: string, detail: Record<string, unknown>, explanation: string) {
  return { id, label, score: r(score), present, severity_hint, detail, explanation };
}
function softDist(n: number, peak: number, rnd: () => number): number[] {
  const v = Array.from({ length: n }, (_, i) => Math.exp(-Math.abs(i - peak)) + rnd() * 0.1);
  const s = v.reduce((a, c) => a + c, 0);
  return v.map((x) => x / s);
}
function distObj(labels: string[], dist: number[]) { return Object.fromEntries(labels.map((l, i) => [l, r(dist[i])])); }
function argmax(a: number[]) { return a.indexOf(Math.max(...a)); }
function pctNum(x: number) { return `${Math.round(x * 100)}%`; }

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
