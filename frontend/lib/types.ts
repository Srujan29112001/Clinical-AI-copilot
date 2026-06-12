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
}

export interface ChatHandlers {
  onToken?: (t: string) => void;
  onDone?: () => void;
  onError?: (msg: string) => void;
}

export interface GraphNode {
  id: string; label: string; group: string;
  title?: string; category?: string; severity?: string;
  semantic?: string; drug_class?: string; dose?: string; brands?: string[];
}
export interface GraphEdge { source: string; target: string; label: string }
export interface KnowledgeGraph { nodes: GraphNode[]; edges: GraphEdge[]; groups: string[] }
