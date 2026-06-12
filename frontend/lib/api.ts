/**
 * API client — connects the UI to the FastAPI backend over Server-Sent Events,
 * and transparently falls back to local simulation when no backend is configured
 * (so the public Vercel deployment is fully testable). Set NEXT_PUBLIC_API_URL
 * to your backend origin to switch to the real multi-agent pipeline.
 */
import { buildResult, simulateStream } from "./simulate";
import type {
  AgentEvent, AnalysisResult, ChatHandlers, KnowledgeGraph, LLMConfig, PatientContext,
} from "./types";
import { SAMPLES } from "./samples";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "").replace(/\/$/, "");
export const isLive = () => API_URL.length > 0;

export interface RunHandlers {
  onEvent?: (e: AgentEvent) => void;
  onResult?: (r: AnalysisResult) => void;
  onDone?: () => void;
  onError?: (msg: string) => void;
}

interface RunOptions {
  profile?: "ictal" | "normal" | "sleep";
  datasetId?: string;
  file?: File | null;
  patient: PatientContext;
  llm?: LLMConfig;
}

async function consumeSSE(res: Response, h: RunHandlers) {
  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response stream");
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop() || "";
    for (const part of parts) {
      const line = part.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      try {
        const evt = JSON.parse(line.slice(5).trim()) as AgentEvent;
        h.onEvent?.(evt);
        if (evt.type === "result" && evt.data) h.onResult?.(evt.data as unknown as AnalysisResult);
        if (evt.type === "done") h.onDone?.();
        if (evt.type === "error") h.onError?.(evt.message || "error");
      } catch {
        /* ignore malformed chunk */
      }
    }
  }
}

export async function runAnalysis(opts: RunOptions, h: RunHandlers): Promise<void> {
  // ── Live backend ──
  if (isLive()) {
    try {
      let res: Response;
      if (opts.file) {
        const fd = new FormData();
        fd.append("file", opts.file);
        fd.append("patient", JSON.stringify(opts.patient));
        if (opts.llm) fd.append("llm_config", JSON.stringify(opts.llm));
        res = await fetch(`${API_URL}/api/analyze`, { method: "POST", body: fd });
      } else {
        res = await fetch(`${API_URL}/api/analyze/sample`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ dataset_id: opts.datasetId || "ictal", patient: opts.patient, llm: opts.llm || {} }),
        });
      }
      if (!res.ok) throw new Error(`Backend ${res.status}`);
      await consumeSSE(res, h);
      return;
    } catch (e) {
      h.onEvent?.({ type: "log", message: `Backend unavailable (${(e as Error).message}); using offline demo.` });
      // fall through to simulation
    }
  }

  // ── Offline simulation ──
  const profile = opts.profile || (SAMPLES.find((s) => s.id === opts.datasetId)?.profile) || "upload";
  const recording = opts.file
    ? { source: opts.file.name, size_kb: Math.round(opts.file.size / 1024), channels: 16 }
    : undefined;
  try {
    for await (const evt of simulateStream(profile as never, opts.patient, recording)) {
      h.onEvent?.(evt);
      if (evt.type === "result" && evt.data) h.onResult?.(evt.data as unknown as AnalysisResult);
      if (evt.type === "done") h.onDone?.();
    }
  } catch (e) {
    h.onError?.((e as Error).message);
  }
}

export async function streamChat(
  messages: Array<{ role: string; content: string }>,
  context: { patient?: PatientContext; analysis?: AnalysisResult | null },
  llm: LLMConfig | undefined,
  h: ChatHandlers,
): Promise<void> {
  if (isLive()) {
    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages, patient: context.patient, analysis: context.analysis, llm: llm || {} }),
      });
      if (!res.ok) throw new Error(`Backend ${res.status}`);
      const reader = res.body!.getReader();
      const dec = new TextDecoder();
      let buf = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() || "";
        for (const part of parts) {
          const line = part.split("\n").find((l) => l.startsWith("data:"));
          if (!line) continue;
          try {
            const evt = JSON.parse(line.slice(5).trim());
            if (evt.type === "token") h.onToken?.(evt.message || "");
            if (evt.type === "done") h.onDone?.();
          } catch { /* ignore */ }
        }
      }
      return;
    } catch {
      /* fall through */
    }
  }
  // offline: synthesize an answer from context
  const reply = offlineChat(messages, context);
  for (const word of reply.split(" ")) {
    h.onToken?.(word + " ");
    await new Promise((r) => setTimeout(r, 18));
  }
  h.onDone?.();
}

function offlineChat(messages: Array<{ role: string; content: string }>, ctx: { analysis?: AnalysisResult | null }): string {
  const q = (messages[messages.length - 1]?.content || "").toLowerCase();
  const a = ctx.analysis;
  if (a) {
    if (/(seizure|epilep|ictal|risk)/.test(q))
      return `Based on the current analysis, the seizure probability is ${(a.eeg.seizure_probability * 100).toFixed(0)}% and the disposition is ${a.urgency}. Key drivers: ${a.eeg.seizure_reasons.join("; ")}. ${a.urgency === "Emergent" ? "Treat this as time-critical and involve neurology immediately." : "Correlate with clinical findings."} (Decision-support only.)`;
    if (/(drug|medic|interaction)/.test(q))
      return a.drug_safety.count ? `I found ${a.drug_safety.count} interaction(s); the highest severity is ${a.drug_safety.highest_severity}. ${a.drug_safety.interactions.map((i) => `${i.drug1} + ${i.drug2}: ${i.description} → ${i.management}`).join(". ")}.` : "No significant drug–drug interactions were detected in the current medication list.";
    if (/(diagnos|condition|icd)/.test(q))
      return `The leading impression is ${a.primary_diagnosis.condition} (ICD-10 ${a.primary_diagnosis.icd10}) at ${(a.primary_diagnosis.confidence * 100).toFixed(0)}% confidence. ${a.primary_diagnosis.reasoning}`;
    if (/(treat|manage|drug|medication|therapy)/.test(q))
      return `Suggested options: ${a.treatments.map((t) => `${t.name} (${t.dosage})`).join(", ")}. Always confirm against contraindications and the drug-safety check.`;
  }
  return "I'm the Clinical AI Copilot assistant. Run an analysis in the Studio and I can explain the EEG findings, the diagnosis reasoning, drug-safety checks, or the knowledge graph behind a result. Connect a local model (Ollama/vLLM) or an API key in Settings for full conversational reasoning.";
}

export async function fetchKnowledgeGraph(): Promise<KnowledgeGraph> {
  if (isLive()) {
    try {
      const res = await fetch(`${API_URL}/api/knowledge-graph`);
      if (res.ok) return (await res.json()) as KnowledgeGraph;
    } catch { /* fall through */ }
  }
  const { STATIC_GRAPH } = await import("./knowledge-data");
  return STATIC_GRAPH;
}

export function quickResult(profile: "ictal" | "normal" | "sleep"): AnalysisResult {
  const s = SAMPLES.find((x) => x.profile === profile)!;
  return buildResult(profile, s.patient);
}
