export type AgentStatus = "pending" | "running" | "done" | "error";

export interface AgentDef {
  id: string;
  name: string;
  role: string;
  icon: string;
}

export interface AgentEvent {
  type: "stage" | "log" | "token" | "result" | "error" | "done";
  agent?: string;
  status?: AgentStatus;
  message?: string;
  data?: Record<string, unknown>;
}

export interface PatientContext {
  patient_id: string;
  age?: number;
  sex?: string;
  symptoms: string;
  history?: string;
  medications: string[];
}

export interface LLMConfig {
  provider?: string;
  model?: string;
  api_key?: string;
  base_url?: string;
  temperature?: number;
  max_tokens?: number;
}

export interface BandPowers {
  delta: number; theta: number; alpha: number; beta: number; gamma: number;
}

export interface AnalysisResult {
  report_id: string;
  patient_id: string;
  urgency: string;
  severity: string;
  summary: string;
  disclaimer: string;
  primary_diagnosis: {
    condition: string;
    icd10?: string;
    confidence: number;
    reasoning: string;
    provider?: string;
  };
  diagnoses: Array<{
    icd10: string; condition: string; category: string;
    eeg_markers: string[]; severity: string; confidence: number;
  }>;
  treatments: Array<{ name: string; type: string; dosage: string; line: string }>;
  evidence: Array<{ source: string; snippet: string; score: number }>;
  triage: { urgency: string; narrative: string; provider?: string };
  eeg: {
    band_powers: BandPowers;
    dominant_band: string;
    theta_beta_ratio: number;
    spectral_entropy: number;
    synchrony: number;
    line_length: number;
    seizure_probability: number;
    seizure_reasons: string[];
    psd_preview: Array<{ freq: number; power: number }>;
    waveform_preview: Array<{ t: number; v: number }>;
    recording: Record<string, unknown>;
  };
  drug_safety: {
    medications: string[];
    interactions: Array<{
      drug1: string; drug2: string; severity: string;
      description: string; mechanism: string; management: string;
    }>;
    count: number;
    highest_severity: string | null;
    requires_action: boolean;
  };
  safety: { passed: boolean; notes: string[]; calibrated_confidence: number };

  // ── v3 additions (optional for backward-compat) ──
  confidence?: number;
  uncertainty?: number;
  primary_finding?: string;
  models_used?: Record<string, string>;
  qeeg?: QEEG;
  detections?: Detection[];
  neuromorphic?: Neuromorphic;
  temporal?: Temporal;
  fusion?: Fusion;
}

export interface QEEG {
  n_features: number;
  channels: string[];
  spectral: {
    rel_band_power: BandPowers;
    abs_band_power: BandPowers;
    band_ratios: Record<string, number>;
    sef95: number; median_freq: number; peak_freq: number; spectral_entropy: number;
  };
  time_domain: Record<string, number>;
  complexity: Record<string, number>;
  connectivity: { mean_correlation: number; alpha_coherence: number; plv: number; pli: number };
  asymmetry: { index: number; lateralization: string; per_band: Record<string, number> };
  events: { spike_count: number; spike_rate_hz: number };
  states: { suppression_ratio: number; burst_suppression: boolean };
  quality: Record<string, number>;
  topography: Array<{ channel: string; rel_band_power: BandPowers; dominant: string }>;
}

export interface Detection {
  id: string; label: string; score: number; present: boolean;
  severity_hint: string; detail: Record<string, unknown>; explanation: string;
}

export interface Neuromorphic {
  top_events: Array<{ event: string; activation: number; spikes: number }>;
  detected_event: string; firing_rate: number; total_spikes: number; energy_uj: number;
  raster: number[][]; raster_labels: string[]; steps: number; neurons: number;
}

export interface Temporal {
  state_norm_series: number[]; trend_slope: number; trend: string; windows: number;
}

export interface Fusion {
  severity: string; severity_dist: Record<string, number>;
  urgency: string; urgency_dist: Record<string, number>;
  confidence: number; uncertainty: number; attention_to_text: number; mc_samples: number;
}

export interface ProviderInfo {
  id: string; default_model: string; models: string[]; default_base_url: string;
  local: boolean; needs_key: boolean; kind: string;
}

export interface LocalModelsResult {
  reachable: boolean; base_url: string; count: number;
  models: Array<{ id: string; size_gb?: number; params?: string; quant?: string; server: string }>;
}

export interface ChatHandlers {
  onToken?: (t: string) => void;
  onDone?: () => void;
  onError?: (msg: string) => void;
}

export interface GraphNode {
  id: string; label: string; group: string;
  description?: string;
  title?: string; category?: string; severity?: string;
  semantic?: string; drug_class?: string; dose?: string; brands?: string[];
}
export interface GraphEdge { source: string; target: string; label: string }
export interface KnowledgeGraph { nodes: GraphNode[]; edges: GraphEdge[]; groups: string[] }
