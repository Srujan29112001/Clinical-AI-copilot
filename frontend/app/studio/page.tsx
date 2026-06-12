"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Sparkles, FlaskConical } from "lucide-react";
import { InputPanel, type RunInput } from "@/components/studio/input-panel";
import { SettingsPanel } from "@/components/studio/settings-panel";
import { AgentTimeline, type AgentState } from "@/components/studio/agent-timeline";
import { Results } from "@/components/studio/results";
import { runAnalysis, isLive } from "@/lib/api";
import { AGENTS } from "@/lib/agents";
import { useLLMConfig, useLastResult, useLastPatient } from "@/lib/store";
import type { AgentEvent, AnalysisResult } from "@/lib/types";

const blankStates = (): Record<string, AgentState> =>
  Object.fromEntries(AGENTS.map((a) => [a.id, { status: "pending" as const }]));

export default function StudioPage() {
  const [llm, setLlm] = useLLMConfig();
  const [, setStoredResult] = useLastResult();
  const [, setStoredPatient] = useLastPatient();
  const [running, setRunning] = useState(false);
  const [states, setStates] = useState<Record<string, AgentState>>(blankStates());
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [logs, setLogs] = useState<string[]>([]);

  const run = async (input: RunInput) => {
    setRunning(true);
    setResult(null);
    setStates(blankStates());
    setLogs([]);
    setStoredPatient(input.patient);

    await runAnalysis(
      {
        file: input.file,
        datasetId: input.sample?.id,
        profile: input.sample?.profile,
        patient: input.patient,
        llm,
      },
      {
        onEvent: (e: AgentEvent) => {
          if (e.type === "log" && e.message) setLogs((l) => [...l, e.message!]);
          if (e.type === "stage" && e.agent) {
            setStates((s) => ({ ...s, [e.agent!]: { status: e.status || "running", message: e.message || s[e.agent!]?.message } }));
          }
        },
        onResult: (r) => { setResult(r); setStoredResult(r); },
        onDone: () => setRunning(false),
        onError: () => setRunning(false),
      },
    );
    setRunning(false);
  };

  return (
    <div className="mx-auto w-full max-w-[1600px] px-5 py-10 sm:px-8">
      <header className="mb-8">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-3xl font-bold tracking-tight">Clinical Studio</h1>
          <span className="chip">
            <span className={`h-1.5 w-1.5 rounded-full ${isLive() ? "bg-emerald-400" : "bg-amber-400"} animate-pulse-glow`} />
            {isLive() ? "Live backend" : "Offline demo"}
          </span>
        </div>
        <p className="mt-2 max-w-2xl text-[var(--color-muted)]">
          Upload an EEG/clinical dataset or pick a sample, then watch the eleven-agent pipeline
          produce a triaged, evidence-backed clinical report in real time.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[330px_minmax(0,1fr)]">
        {/* left: controls (sticky on large screens) */}
        <aside className="space-y-5 lg:sticky lg:top-20 lg:self-start">
          <SettingsPanel config={llm} onChange={setLlm} />
          <div className="panel p-5">
            <InputPanel running={running} onRun={run} />
          </div>
        </aside>

        {/* right: pipeline + full-width results */}
        <main className="min-w-0 space-y-6">
          {(running || result) && (
            <div className="panel p-5">
              <div className="mb-4 flex items-center justify-between">
                <h3 className="flex items-center gap-2 text-sm font-semibold">
                  <Sparkles className="h-4 w-4 text-[var(--color-cyan)]" /> Agent pipeline
                </h3>
                {running && <span className="chip !py-0.5 text-[var(--color-cyan)]">running…</span>}
              </div>
              <AgentTimeline states={states} variant="grid" />
              <AnimatePresence>
                {logs.length > 0 && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-4 max-h-28 overflow-y-auto rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-soft)] p-3">
                    <p className="mb-1 text-[10px] uppercase tracking-wider text-[var(--color-faint)]">System log</p>
                    {logs.map((l, i) => (
                      <p key={i} className="break-words font-mono text-[11px] text-[var(--color-muted)]">› {l}</p>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}

          {!result && !running && <EmptyState />}
          {result && <Results result={result} />}
        </main>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="panel flex min-h-[420px] flex-col items-center justify-center p-10 text-center">
      <FlaskConical className="h-10 w-10 text-[var(--color-faint)]" />
      <h3 className="mt-4 text-lg font-semibold">No analysis yet</h3>
      <p className="mt-2 max-w-sm text-sm text-[var(--color-muted)]">
        Choose a sample dataset or upload your own EEG file on the left, set patient context, and
        run the pipeline. The live agent pipeline and full report — diagnosis, EEG charts, spike
        raster, detectors, treatments and safety — appear here.
      </p>
    </div>
  );
}
