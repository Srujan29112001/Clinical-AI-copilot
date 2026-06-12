"use client";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  AlertTriangle, Stethoscope, Activity, Pill, BookOpen, ShieldCheck, FileText,
  MessageSquare, TriangleAlert,
} from "lucide-react";
import type { AnalysisResult } from "@/lib/types";
import { URGENCY_STYLES, SEVERITY_BY_INTERACTION, cn } from "@/lib/utils";
import { WaveformChart, PSDChart, BandPowerBars, Donut } from "./charts";

export function Results({ result }: { result: AnalysisResult }) {
  const u = URGENCY_STYLES[result.urgency] || URGENCY_STYLES.Routine;
  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="space-y-5">
      {/* urgency banner */}
      <div className={cn("panel flex flex-wrap items-center justify-between gap-4 p-5 ring-1", u.bg, u.ring)}>
        <div className="flex items-center gap-3">
          <AlertTriangle className={cn("h-6 w-6", u.text)} />
          <div>
            <p className="text-xs uppercase tracking-wider text-[var(--color-muted)]">Disposition</p>
            <p className={cn("text-xl font-bold", u.text)}>{result.urgency}</p>
          </div>
        </div>
        <div className="flex items-center gap-6 text-sm">
          <Meta label="Severity" value={result.severity} />
          <Meta label="Report" value={result.report_id} mono />
          <Meta label="Patient" value={result.patient_id} mono />
        </div>
        <Link href="/chat" className="btn btn-ghost !py-2">
          <MessageSquare className="h-4 w-4" /> Ask the copilot
        </Link>
      </div>

      <div className="grid gap-5 lg:grid-cols-3">
        {/* primary diagnosis */}
        <Section title="Primary diagnosis" icon={Stethoscope} className="lg:col-span-2">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
            <Donut value={result.primary_diagnosis.confidence} label="confidence" color="var(--color-violet)" />
            <div className="flex-1">
              <h3 className="text-lg font-semibold">{result.primary_diagnosis.condition}</h3>
              {result.primary_diagnosis.icd10 && (
                <span className="chip mt-1">ICD-10 · {result.primary_diagnosis.icd10}</span>
              )}
              <p className="mt-3 text-sm leading-relaxed text-[var(--color-muted)]">
                {result.primary_diagnosis.reasoning}
              </p>
              {result.primary_diagnosis.provider && result.primary_diagnosis.provider !== "simulated" && (
                <p className="mt-2 text-xs text-[var(--color-faint)]">Reasoned by: {result.primary_diagnosis.provider}</p>
              )}
            </div>
          </div>
        </Section>

        {/* seizure probability */}
        <Section title="Seizure risk" icon={Activity}>
          <div className="flex items-center gap-4">
            <Donut
              value={result.eeg.seizure_probability}
              label="probability"
              color={result.eeg.seizure_probability >= 0.8 ? "var(--color-red)" : result.eeg.seizure_probability >= 0.45 ? "var(--color-amber)" : "var(--color-green)"}
            />
            <ul className="flex-1 space-y-1.5 text-xs text-[var(--color-muted)]">
              {result.eeg.seizure_reasons.map((r, i) => (
                <li key={i} className="flex gap-2"><span className="text-[var(--color-cyan)]">›</span> {r}</li>
              ))}
            </ul>
          </div>
        </Section>
      </div>

      {/* EEG analysis */}
      <Section title="EEG signal analysis" icon={Activity}>
        <div className="grid gap-6 lg:grid-cols-2">
          <div>
            <Label>Waveform (lead 1)</Label>
            <WaveformChart data={result.eeg.waveform_preview} />
            <Label className="mt-4">Power spectral density (0–50 Hz)</Label>
            <PSDChart data={result.eeg.psd_preview} />
          </div>
          <div>
            <Label>Relative band power</Label>
            <div className="mt-2"><BandPowerBars bands={result.eeg.band_powers} /></div>
            <div className="mt-4 grid grid-cols-3 gap-3">
              <Stat label="Dominant" value={result.eeg.dominant_band} />
              <Stat label="θ/β ratio" value={result.eeg.theta_beta_ratio.toFixed(2)} />
              <Stat label="Synchrony" value={result.eeg.synchrony.toFixed(2)} />
              <Stat label="Entropy" value={result.eeg.spectral_entropy.toFixed(2)} />
              <Stat label="Line length" value={result.eeg.line_length.toFixed(0)} />
              <Stat label="Channels" value={String(result.eeg.recording?.channels ?? "—")} />
            </div>
          </div>
        </div>
      </Section>

      <div className="grid gap-5 lg:grid-cols-2">
        {/* treatments */}
        <Section title="Recommended treatments" icon={Pill}>
          <ul className="space-y-2">
            {result.treatments.map((t, i) => (
              <li key={i} className="flex items-center justify-between rounded-lg border border-[var(--color-border)] px-3 py-2.5">
                <div>
                  <span className="text-sm font-medium">{t.name}</span>
                  <span className="ml-2 text-xs text-[var(--color-faint)]">{t.type}</span>
                </div>
                <div className="text-right">
                  <span className="text-xs text-[var(--color-muted)]">{t.dosage}</span>
                  <span className={cn("ml-2 chip !py-0.5 !text-[10px]", t.line === "first" ? "!text-emerald-300" : "")}>{t.line}-line</span>
                </div>
              </li>
            ))}
          </ul>
        </Section>

        {/* drug safety */}
        <Section title="Drug-safety check" icon={ShieldCheck}>
          {result.drug_safety.count === 0 ? (
            <p className="rounded-lg border border-emerald-500/30 bg-emerald-500/[0.06] px-3 py-3 text-sm text-emerald-300">
              No significant drug–drug interactions detected
              {result.drug_safety.medications.length ? ` across ${result.drug_safety.medications.length} medication(s).` : "."}
            </p>
          ) : (
            <ul className="space-y-2">
              {result.drug_safety.interactions.map((it, i) => (
                <li key={i} className={cn("rounded-lg border px-3 py-2.5 text-sm", SEVERITY_BY_INTERACTION[it.severity])}>
                  <div className="flex items-center gap-2 font-medium">
                    <TriangleAlert className="h-4 w-4" /> {it.drug1} + {it.drug2}
                    <span className="ml-auto text-[10px] uppercase tracking-wider">{it.severity}</span>
                  </div>
                  <p className="mt-1 text-xs opacity-90">{it.description}</p>
                  <p className="mt-0.5 text-xs opacity-75"><strong>Manage:</strong> {it.management}</p>
                </li>
              ))}
            </ul>
          )}
        </Section>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        {/* differentials */}
        <Section title="Differential diagnoses" icon={Stethoscope}>
          <ul className="space-y-2">
            {result.diagnoses.map((d, i) => (
              <li key={i} className="rounded-lg border border-[var(--color-border)] px-3 py-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium line-clamp-1">{d.condition}</span>
                  <span className="font-mono text-xs text-[var(--color-cyan)]">{(d.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="mt-1 flex flex-wrap gap-1.5">
                  <span className="chip !py-0.5 !text-[10px]">{d.icd10}</span>
                  {d.eeg_markers.slice(0, 2).map((m) => (
                    <span key={m} className="chip !py-0.5 !text-[10px] !text-amber-300/80">{m}</span>
                  ))}
                </div>
              </li>
            ))}
          </ul>
        </Section>

        {/* evidence */}
        <Section title="Supporting evidence" icon={BookOpen}>
          <ul className="space-y-2">
            {result.evidence.map((e, i) => (
              <li key={i} className="rounded-lg border border-[var(--color-border)] px-3 py-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-[var(--color-cyan)]">{e.source}</span>
                  <span className="font-mono text-xs text-[var(--color-faint)]">{(e.score * 100).toFixed(0)}%</span>
                </div>
                <p className="mt-1 text-xs text-[var(--color-muted)]">{e.snippet}</p>
              </li>
            ))}
          </ul>
        </Section>
      </div>

      {/* safety + summary */}
      <Section title="Safety review & clinical summary" icon={FileText}>
        {result.safety.notes.length > 0 ? (
          <ul className="mb-4 space-y-1.5">
            {result.safety.notes.map((n, i) => (
              <li key={i} className="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/[0.06] px-3 py-2 text-sm text-amber-200">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" /> {n}
              </li>
            ))}
          </ul>
        ) : (
          <p className="mb-4 rounded-lg border border-emerald-500/30 bg-emerald-500/[0.06] px-3 py-2 text-sm text-emerald-300">
            All guardrail checks passed · calibrated confidence {(result.safety.calibrated_confidence * 100).toFixed(0)}%
          </p>
        )}
        <p className="text-sm leading-relaxed text-[var(--color-muted)]">{result.summary}</p>
        <p className="mt-4 border-t border-[var(--color-border)] pt-3 text-xs text-[var(--color-faint)]">⚠ {result.disclaimer}</p>
      </Section>
    </motion.div>
  );
}

function Section({ title, icon: Icon, children, className }: { title: string; icon: typeof Activity; children: React.ReactNode; className?: string }) {
  return (
    <div className={cn("panel p-5", className)}>
      <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold">
        <Icon className="h-4 w-4 text-[var(--color-cyan)]" /> {title}
      </h3>
      {children}
    </div>
  );
}

function Meta({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <p className="text-xs uppercase tracking-wider text-[var(--color-faint)]">{label}</p>
      <p className={cn("font-semibold capitalize", mono && "font-mono text-xs normal-case")}>{value}</p>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-[var(--color-border)] px-2.5 py-2 text-center">
      <p className="text-sm font-bold capitalize text-[var(--color-ink)]">{value}</p>
      <p className="text-[10px] text-[var(--color-faint)]">{label}</p>
    </div>
  );
}

function Label({ children, className }: { children: React.ReactNode; className?: string }) {
  return <p className={cn("text-xs font-medium text-[var(--color-muted)]", className)}>{children}</p>;
}
