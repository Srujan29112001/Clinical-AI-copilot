"use client";
import Link from "next/link";
import { ArrowRight, Boxes, GitBranch, Server, Brain, ShieldCheck, Zap, Code2 } from "lucide-react";
import { Reveal } from "@/components/ui/reveal";
import { AGENTS } from "@/lib/agents";
import { AgentIcon } from "@/components/ui/agent-icon";

const ENDPOINTS = [
  ["POST", "/api/analyze", "Multipart upload → SSE multi-agent run"],
  ["POST", "/api/analyze/sample", "JSON {dataset_id} → SSE run"],
  ["POST", "/api/chat", "SSE token stream from the copilot"],
  ["GET", "/api/knowledge-graph", "ICD-10 / SNOMED / RxNorm graph"],
  ["GET", "/api/providers", "LLM provider catalog (local + hosted)"],
  ["GET", "/api/agents", "Multi-agent roster"],
  ["GET", "/api/samples", "Built-in demo datasets"],
  ["GET", "/health", "Liveness + provider status"],
];

const UPGRADE = [
  ["Interface", "Backend-only Python, no UI", "Next.js 15 + React 19 web app, deployable to Vercel"],
  ["Inference", "Hard-wired Llama 3.1 on a 12 GB GPU", "Hybrid layer — local GPU (Ollama/vLLM) or any API, per agent"],
  ["Orchestration", "Monolithic request handler", "7-agent streaming pipeline with a safety critic"],
  ["Knowledge", "Neo4j + Qdrant (heavy infra)", "Portable in-memory GraphRAG over the same ontologies"],
  ["Testability", "Required full Docker stack", "Zero-key live demo runs in the browser"],
  ["Signal stack", "MNE + CUDA dependencies", "NumPy/SciPy pipeline runs anywhere, GPU optional"],
];

export default function ArchitecturePage() {
  return (
    <div className="container-page py-12">
      <header className="mx-auto max-w-3xl text-center">
        <span className="chip mx-auto">System dossier</span>
        <h1 className="mt-4 text-4xl font-bold tracking-tight sm:text-5xl">Architecture & upgrade</h1>
        <p className="mt-4 text-[var(--color-muted)]">
          How a backend-only EEG research repo became a deployable, multi-agent, hybrid-inference
          clinical product — without throwing away the deep-learning engine underneath.
        </p>
      </header>

      {/* upgrade table */}
      <section className="mx-auto mt-14 max-w-4xl">
        <h2 className="mb-5 flex items-center gap-2 text-2xl font-bold"><Boxes className="h-6 w-6 text-[var(--color-cyan)]" /> What changed</h2>
        <div className="panel divide-y divide-[var(--color-border)]">
          {UPGRADE.map(([dim, before, after], i) => (
            <Reveal key={dim} delay={i * 0.05}>
              <div className="grid gap-3 p-4 sm:grid-cols-[140px_1fr_1fr] sm:items-center">
                <span className="text-sm font-semibold text-[var(--color-cyan)]">{dim}</span>
                <span className="text-sm text-[var(--color-faint)] line-through decoration-[var(--color-rose)]/40">{before}</span>
                <span className="flex items-center gap-2 text-sm text-[var(--color-ink)]"><ArrowRight className="h-3.5 w-3.5 text-[var(--color-teal)]" /> {after}</span>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* data flow */}
      <section className="mx-auto mt-16 max-w-4xl">
        <h2 className="mb-5 flex items-center gap-2 text-2xl font-bold"><GitBranch className="h-6 w-6 text-[var(--color-cyan)]" /> Request flow</h2>
        <div className="panel p-6">
          <pre className="overflow-x-auto font-mono text-xs leading-relaxed text-[var(--color-muted)]">
{`Browser (Next.js)                         FastAPI backend
─────────────────                         ───────────────
upload EEG ──► POST /api/analyze ─────────► load_signal()  (CSV/EDF/NPY/JSON)
                  (multipart + SSE)              │
                                                 ▼
   ◄──── data: {stage:"signal",running} ─── Signal Analyst  (filter, FFT, features)
   ◄──── data: {stage:"triage",...} ─────── Triage          (urgency)
   ◄──── data: {stage:"retriever",...} ──── Knowledge RAG   (ICD-10/SNOMED/RxNorm)
   ◄──── data: {stage:"diagnostician"} ──── Diagnostician   ┐
   ◄──── data: {stage:"pharmacologist"} ─── Pharmacologist  ├─ each on its own LLM
   ◄──── data: {stage:"critic",...} ─────── Safety Critic   ┘  (local GPU OR API)
   ◄──── data: {stage:"reporter",...} ───── Reporter
   ◄──── data: {type:"result", ...}  ────── final report
   ◄──── data: {type:"done"}

No backend?  →  the same event shape is generated client-side (offline demo).`}
          </pre>
        </div>
      </section>

      {/* agents */}
      <section className="mx-auto mt-16 max-w-4xl">
        <h2 className="mb-5 flex items-center gap-2 text-2xl font-bold"><Brain className="h-6 w-6 text-[var(--color-cyan)]" /> The agents</h2>
        <div className="grid gap-3 sm:grid-cols-2">
          {AGENTS.map((a, i) => (
            <Reveal key={a.id} delay={i * 0.04}>
              <div className="panel-soft flex items-start gap-3 p-4">
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[var(--color-cyan)]/10 ring-1 ring-[var(--color-border)]">
                  <AgentIcon name={a.icon} className="h-4 w-4 text-[var(--color-cyan)]" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold">{i + 1}. {a.name}</h3>
                  <p className="text-xs text-[var(--color-muted)]">{a.role}</p>
                </div>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* API */}
      <section id="api" className="mx-auto mt-16 max-w-4xl">
        <h2 className="mb-5 flex items-center gap-2 text-2xl font-bold"><Server className="h-6 w-6 text-[var(--color-cyan)]" /> API surface</h2>
        <div className="panel divide-y divide-[var(--color-border)]">
          {ENDPOINTS.map(([m, path, desc]) => (
            <div key={path} className="flex flex-wrap items-center gap-3 p-3.5">
              <span className={`w-14 rounded-md px-2 py-0.5 text-center font-mono text-xs ${m === "GET" ? "bg-emerald-500/10 text-emerald-300" : "bg-rose-500/10 text-rose-300"}`}>{m}</span>
              <code className="font-mono text-sm text-[var(--color-ink)]">{path}</code>
              <span className="text-sm text-[var(--color-faint)]">{desc}</span>
            </div>
          ))}
        </div>
      </section>

      {/* stack + research engine */}
      <section className="mx-auto mt-16 grid max-w-4xl gap-5 sm:grid-cols-2">
        <Card icon={Zap} title="Portable runtime" items={["FastAPI + SSE", "NumPy/SciPy DSP", "In-memory GraphRAG", "Deterministic demo engine", "No GPU required to run"]} />
        <Card icon={ShieldCheck} title="Deep-learning engine (src/)" items={["CNN-LSTM seizure detector", "Transformer sleep stager", "Mamba2 long-context SSM", "Llama 3.1 8B QLoRA", "Spiking NN event detector"]} />
      </section>

      <section className="mx-auto mt-16 max-w-4xl">
        <div className="panel flex flex-col items-center gap-4 p-8 text-center sm:flex-row sm:justify-between sm:text-left">
          <div>
            <h3 className="flex items-center gap-2 text-lg font-bold"><Code2 className="h-5 w-5 text-[var(--color-cyan)]" /> Run it locally</h3>
            <p className="mt-1 text-sm text-[var(--color-muted)]">Backend: <code className="font-mono">uvicorn app.main:app</code> · Frontend: <code className="font-mono">npm run dev</code></p>
          </div>
          <Link href="/studio" className="btn btn-primary">Open the Studio <ArrowRight className="h-4 w-4" /></Link>
        </div>
      </section>
    </div>
  );
}

function Card({ icon: Icon, title, items }: { icon: typeof Zap; title: string; items: string[] }) {
  return (
    <div className="panel p-6">
      <Icon className="h-6 w-6 text-[var(--color-cyan)]" />
      <h3 className="mt-3 font-semibold">{title}</h3>
      <ul className="mt-3 space-y-1.5">
        {items.map((it) => (
          <li key={it} className="flex items-center gap-2 text-sm text-[var(--color-muted)]"><span className="h-1 w-1 rounded-full bg-[var(--color-teal)]" /> {it}</li>
        ))}
      </ul>
    </div>
  );
}
