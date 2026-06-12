"use client";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  Cpu, Cloud, Server, ArrowRight, ShieldCheck, Zap, GitBranch, Database, Brain,
} from "lucide-react";
import { AGENTS } from "@/lib/agents";
import { AgentIcon } from "@/components/ui/agent-icon";
import { Reveal, Stagger, StaggerItem } from "@/components/ui/reveal";

function Heading({ eyebrow, title, sub }: { eyebrow: string; title: string; sub?: string }) {
  return (
    <div className="mx-auto max-w-2xl text-center">
      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[var(--color-cyan)]">{eyebrow}</p>
      <h2 className="mt-3 text-3xl font-bold tracking-tight sm:text-4xl">{title}</h2>
      {sub && <p className="mt-4 text-[var(--color-muted)]">{sub}</p>}
    </div>
  );
}

/* ── Stats band ── */
export function Stats() {
  const items = [
    ["7", "specialized agents"],
    ["11", "LLM providers"],
    ["<2s", "demo latency"],
    ["100%", "key-free demo"],
  ];
  return (
    <section className="container-page py-14">
      <div className="panel grid grid-cols-2 divide-x divide-[var(--color-border)] sm:grid-cols-4">
        {items.map(([v, l], i) => (
          <Reveal key={l} delay={i * 0.08} className="px-4 py-8 text-center">
            <div className="text-4xl font-extrabold gradient-text">{v}</div>
            <div className="mt-1 text-sm text-[var(--color-muted)]">{l}</div>
          </Reveal>
        ))}
      </div>
    </section>
  );
}

/* ── Agent roster ── */
export function AgentRoster() {
  return (
    <section id="agents" className="container-page py-20">
      <Heading
        eyebrow="The crew"
        title="Seven agents, one diagnosis"
        sub="Each agent owns one clinical responsibility and can run on its own model — fast triage on a local GPU, heavy reasoning on a frontier API."
      />
      <Stagger className="mt-12 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {AGENTS.map((a, i) => (
          <StaggerItem key={a.id}>
            <div className="panel-soft group h-full p-6 transition-all hover:-translate-y-1 hover:border-[var(--color-cyan)]/40">
              <div className="flex items-center gap-3">
                <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-[var(--color-cyan)]/20 to-[var(--color-violet)]/20 ring-1 ring-[var(--color-border)]">
                  <AgentIcon name={a.icon} className="h-5 w-5 text-[var(--color-cyan)]" />
                </div>
                <div>
                  <h3 className="font-semibold">{a.name}</h3>
                  <p className="text-xs text-[var(--color-faint)]">Agent {String(i + 1).padStart(2, "0")}</p>
                </div>
              </div>
              <p className="mt-4 text-sm text-[var(--color-muted)]">{a.role}</p>
            </div>
          </StaggerItem>
        ))}
        <StaggerItem>
          <Link href="/studio" className="panel-soft flex h-full flex-col justify-between p-6 transition-all hover:-translate-y-1 hover:border-[var(--color-cyan)]/40">
            <Brain className="h-7 w-7 text-[var(--color-violet)]" />
            <div className="mt-6">
              <h3 className="font-semibold">See them work →</h3>
              <p className="mt-1 text-sm text-[var(--color-muted)]">Run the live pipeline in the Studio.</p>
            </div>
          </Link>
        </StaggerItem>
      </Stagger>
    </section>
  );
}

/* ── Pipeline flow ── */
export function Pipeline() {
  const steps = [
    { t: "Acquire", d: "EDF · CSV · NPY · JSON upload", icon: Database },
    { t: "Preprocess", d: "Filter · artefact removal · features", icon: Zap },
    { t: "Reason", d: "GraphRAG + multi-agent diagnosis", icon: Brain },
    { t: "Report", d: "Triage · treatment · safety check", icon: ShieldCheck },
  ];
  return (
    <section className="container-page py-20">
      <Heading eyebrow="How it flows" title="A self-correcting clinical pipeline" />
      <div className="relative mt-14">
        <div className="absolute left-0 right-0 top-7 hidden h-px bg-gradient-to-r from-transparent via-[var(--color-cyan)]/40 to-transparent md:block" />
        <div className="grid gap-8 md:grid-cols-4">
          {steps.map((s, i) => (
            <Reveal key={s.t} delay={i * 0.1} className="relative text-center">
              <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-[var(--color-panel)] ring-1 ring-[var(--color-border)]">
                <s.icon className="h-6 w-6 text-[var(--color-cyan)]" />
              </div>
              <h3 className="mt-4 font-semibold">{s.t}</h3>
              <p className="mt-1 text-sm text-[var(--color-muted)]">{s.d}</p>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ── Hybrid inference highlight (core requirement) ── */
export function HybridInference() {
  return (
    <section className="container-page py-20">
      <div className="panel relative overflow-hidden p-8 sm:p-12">
        <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-[var(--color-violet)]/10 blur-3xl" />
        <div className="grid items-center gap-10 lg:grid-cols-2">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[var(--color-teal)]">Hybrid inference</p>
            <h2 className="mt-3 text-3xl font-bold tracking-tight">Your GPU or the cloud. Per agent.</h2>
            <p className="mt-4 text-[var(--color-muted)]">
              Run everything privately on a local model with <strong className="text-[var(--color-ink)]">Ollama</strong> or
              {" "}<strong className="text-[var(--color-ink)]">vLLM</strong> — no key, no data leaving the building. Or wire any
              agent to <strong className="text-[var(--color-ink)]">Anthropic, OpenAI, Groq, DeepSeek, Mistral, Gemini</strong> and
              more. Mix and match: cheap fast triage locally, frontier reasoning in the cloud.
            </p>
            <div className="mt-6 flex flex-wrap gap-3">
              <Link href="/studio" className="btn btn-primary">Configure providers <ArrowRight className="h-4 w-4" /></Link>
              <Link href="/architecture" className="btn btn-ghost">Read the architecture</Link>
            </div>
          </div>
          <div className="grid gap-4">
            <Card icon={Server} title="Local GPU" tone="teal"
              body="Ollama / vLLM / LM Studio via an OpenAI-compatible endpoint. Private, offline-capable, zero per-token cost." />
            <Card icon={Cloud} title="Hosted API" tone="violet"
              body="Anthropic Claude · OpenAI · Groq · DeepSeek · Mistral · OpenRouter · Gemini — set a key per role." />
            <Card icon={Cpu} title="Zero-key demo" tone="cyan"
              body="No provider configured? A deterministic engine streams a full, realistic analysis so the app always works." />
          </div>
        </div>
      </div>
    </section>
  );
}

function Card({ icon: Icon, title, body, tone }: { icon: typeof Cpu; title: string; body: string; tone: string }) {
  const ring = tone === "teal" ? "ring-[var(--color-teal)]/30" : tone === "violet" ? "ring-[var(--color-violet)]/30" : "ring-[var(--color-cyan)]/30";
  return (
    <div className={`panel-soft flex gap-4 p-5 ring-1 ${ring}`}>
      <Icon className="h-6 w-6 shrink-0 text-[var(--color-cyan)]" />
      <div>
        <h3 className="font-semibold">{title}</h3>
        <p className="mt-1 text-sm text-[var(--color-muted)]">{body}</p>
      </div>
    </div>
  );
}

/* ── Tech stack ── */
export function TechStack() {
  const groups = [
    { icon: GitBranch, title: "Frontend", items: ["Next.js 15", "React 19", "Tailwind v4", "Framer Motion", "TypeScript"] },
    { icon: Server, title: "Backend", items: ["FastAPI", "SSE streaming", "Multi-agent orchestrator", "Hybrid LLM layer", "NumPy / SciPy DSP"] },
    { icon: Brain, title: "Clinical AI", items: ["GraphRAG", "ICD-10 / SNOMED / RxNorm", "Seizure heuristics", "Drug-interaction engine", "MC-dropout calibration"] },
    { icon: ShieldCheck, title: "Research engine", items: ["CNN-LSTM seizure net", "Transformer sleep stager", "Mamba2 SSM", "Llama 3.1 QLoRA", "Spiking NN"] },
  ];
  return (
    <section className="container-page py-20">
      <Heading eyebrow="Under the hood" title="A modern, upgraded stack" sub="Re-architected from a backend-only research repo into a deployable full-stack product." />
      <div className="mt-12 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {groups.map((g, i) => (
          <Reveal key={g.title} delay={i * 0.08}>
            <div className="panel-soft h-full p-6">
              <g.icon className="h-6 w-6 text-[var(--color-cyan)]" />
              <h3 className="mt-4 font-semibold">{g.title}</h3>
              <ul className="mt-3 space-y-2">
                {g.items.map((it) => (
                  <li key={it} className="flex items-center gap-2 text-sm text-[var(--color-muted)]">
                    <span className="h-1 w-1 rounded-full bg-[var(--color-teal)]" /> {it}
                  </li>
                ))}
              </ul>
            </div>
          </Reveal>
        ))}
      </div>
    </section>
  );
}

/* ── CTA ── */
export function CTA() {
  return (
    <section className="container-page py-20">
      <motion.div
        initial={{ opacity: 0, scale: 0.97 }}
        whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6 }}
        className="panel relative overflow-hidden p-10 text-center sm:p-16"
      >
        <div className="absolute inset-0 -z-10 grid-bg opacity-30" />
        <h2 className="mx-auto max-w-2xl text-3xl font-bold tracking-tight sm:text-4xl">
          Test it with your own dataset in under a minute.
        </h2>
        <p className="mx-auto mt-4 max-w-xl text-[var(--color-muted)]">
          Drag in an EEG file or pick a built-in sample. No setup, no key — the full
          multi-agent analysis runs right in your browser.
        </p>
        <Link href="/studio" className="btn btn-primary mt-8 text-base !px-7 !py-3">
          Launch the Studio <ArrowRight className="h-4 w-4" />
        </Link>
      </motion.div>
    </section>
  );
}
