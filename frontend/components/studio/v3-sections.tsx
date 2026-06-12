"use client";
import { useState } from "react";
import { Zap, Scan, Waves, GitMerge, Brain, Cpu, ChevronDown } from "lucide-react";
import type { AnalysisResult, Detection, Neuromorphic, Temporal, Fusion, QEEG } from "@/lib/types";
import { cn } from "@/lib/utils";

const BAND_COLORS: Record<string, string> = {
  delta: "#e11d48", theta: "#ff3b4e", alpha: "#ff7a59", beta: "#f59e0b", gamma: "#fbbf24",
};

/* ── Models used strip ── */
export function ModelsUsed({ models }: { models: Record<string, string> }) {
  const labels: Record<string, string> = {
    signal: "Signal", events: "Events", temporal: "Temporal", knowledge: "Knowledge",
    fusion: "Fusion", reasoning: "Reasoning",
  };
  return (
    <div className="panel p-4">
      <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold"><Cpu className="h-4 w-4 text-[var(--color-cyan)]" /> Model pipeline</h3>
      <div className="flex flex-wrap gap-2">
        {Object.entries(models).map(([k, v]) => (
          <span key={k} className="chip !py-1">
            <span className="text-[var(--color-faint)]">{labels[k] || k}:</span>
            <span className="text-[var(--color-cyan)]">{v}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

/* ── SNN spike raster ── */
export function SpikeRaster({ neuro }: { neuro: Neuromorphic }) {
  const cell = 8;
  return (
    <div className="panel p-5">
      <h3 className="mb-1 flex items-center gap-2 text-sm font-semibold"><Zap className="h-4 w-4 text-[var(--color-amber)]" /> Neuromorphic event detection (SNN)</h3>
      <p className="mb-3 text-xs text-[var(--color-faint)]">
        Detected: <span className="text-[var(--color-amber)]">{neuro.detected_event}</span> · firing rate {neuro.firing_rate.toFixed(2)} · {neuro.total_spikes} spikes · ~{neuro.energy_uj} µJ
      </p>
      <div className="overflow-x-auto">
        <svg width={neuro.steps * cell + 130} height={neuro.neurons * (cell + 2) + 10}>
          {neuro.raster.map((row, i) => (
            <g key={i}>
              <text x={0} y={i * (cell + 2) + cell + 4} className="fill-[var(--color-muted)]" fontSize="9">
                {neuro.raster_labels[i]?.slice(0, 18)}
              </text>
              {row.map((v, t) => (
                <rect key={t} x={125 + t * cell} y={i * (cell + 2) + 4} width={cell - 1} height={cell}
                  rx={1} fill={v ? (i === argmaxRow(neuro.raster) ? "#f59e0b" : "#ff3b4e") : "var(--color-panel-2)"}
                  opacity={v ? 0.95 : 0.5} />
              ))}
            </g>
          ))}
          <text x={125} y={neuro.neurons * (cell + 2) + 9} fontSize="8" className="fill-[var(--color-faint)]">→ time ({neuro.steps} LIF steps)</text>
        </svg>
      </div>
    </div>
  );
}
function argmaxRow(raster: number[][]) {
  const sums = raster.map((r) => r.reduce((a, c) => a + c, 0));
  return sums.indexOf(Math.max(...sums));
}

/* ── 7-detector grid ── */
export function DetectionsGrid({ detections, primary }: { detections: Detection[]; primary?: string }) {
  return (
    <div className="panel p-5">
      <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold"><Scan className="h-4 w-4 text-[var(--color-cyan)]" /> Detection use-cases ({detections.length})</h3>
      <div className="grid gap-2 sm:grid-cols-2">
        {detections.map((d) => (
          <div key={d.id} className={cn("rounded-lg border p-3", d.id === primary ? "border-[var(--color-cyan)] bg-[var(--color-cyan)]/[0.06]" : d.present ? "border-amber-500/30" : "border-[var(--color-border)]")}>
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">{d.label}</span>
              <span className={cn("font-mono text-xs", d.present ? "text-amber-300" : "text-[var(--color-faint)]")}>{Math.round(d.score * 100)}%</span>
            </div>
            <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-[var(--color-bg-soft)]">
              <div className="h-full rounded-full" style={{ width: `${d.score * 100}%`, background: d.present ? "var(--color-amber)" : "var(--color-faint)" }} />
            </div>
            <p className="mt-1.5 text-xs text-[var(--color-faint)] line-clamp-2">{d.explanation}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── Fusion + uncertainty ── */
export function FusionPanel({ fusion }: { fusion: Fusion }) {
  return (
    <div className="panel p-5">
      <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold"><GitMerge className="h-4 w-4 text-[var(--color-violet)]" /> Multimodal fusion (cross-attention)</h3>
      <div className="grid gap-4 sm:grid-cols-2">
        <Dist title="Severity" dist={fusion.severity_dist} pick={fusion.severity} color="#ff5d8f" />
        <Dist title="Urgency" dist={fusion.urgency_dist} pick={fusion.urgency} color="#ff3b4e" />
      </div>
      <div className="mt-4 grid grid-cols-3 gap-3 text-center">
        <Mini label="Confidence" value={`${Math.round(fusion.confidence * 100)}%`} />
        <Mini label="Uncertainty (MC)" value={fusion.uncertainty.toFixed(3)} warn={fusion.uncertainty > 0.12} />
        <Mini label="Attn → text" value={fusion.attention_to_text.toFixed(2)} />
      </div>
      <p className="mt-2 text-center text-[10px] text-[var(--color-faint)]">{fusion.mc_samples} Monte-Carlo dropout passes</p>
    </div>
  );
}
function Dist({ title, dist, pick, color }: { title: string; dist: Record<string, number>; pick: string; color: string }) {
  return (
    <div>
      <p className="mb-1.5 text-xs font-medium text-[var(--color-muted)]">{title}</p>
      <div className="space-y-1">
        {Object.entries(dist).map(([k, v]) => (
          <div key={k} className="flex items-center gap-2">
            <span className={cn("w-16 text-[11px] capitalize", k === pick ? "text-[var(--color-ink)] font-semibold" : "text-[var(--color-faint)]")}>{k}</span>
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[var(--color-bg-soft)]">
              <div className="h-full rounded-full" style={{ width: `${v * 100}%`, background: k === pick ? color : "var(--color-border)" }} />
            </div>
            <span className="w-8 text-right font-mono text-[10px] text-[var(--color-faint)]">{Math.round(v * 100)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
function Mini({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div className="rounded-lg border border-[var(--color-border)] py-2">
      <p className={cn("text-base font-bold", warn ? "text-amber-300" : "text-[var(--color-ink)]")}>{value}</p>
      <p className="text-[10px] text-[var(--color-faint)]">{label}</p>
    </div>
  );
}

/* ── Mamba2 temporal trend ── */
export function TemporalTrend({ temporal }: { temporal: Temporal }) {
  const s = temporal.state_norm_series;
  const max = Math.max(...s, 1), min = Math.min(...s, 0);
  const W = 280, H = 60;
  const pts = s.map((v, i) => `${(i / (s.length - 1)) * W},${H - ((v - min) / (max - min + 1e-9)) * (H - 8) - 4}`).join(" ");
  const color = temporal.trend === "escalating" ? "#fb7185" : temporal.trend === "resolving" ? "#34d399" : "#ff3b4e";
  return (
    <div className="panel p-5">
      <h3 className="mb-1 flex items-center gap-2 text-sm font-semibold"><Waves className="h-4 w-4 text-[var(--color-teal)]" /> Temporal trend (Mamba2 SSM)</h3>
      <p className="mb-3 text-xs text-[var(--color-faint)]">State-space norm across {temporal.windows} windows · trend: <span style={{ color }}>{temporal.trend}</span> (slope {temporal.trend_slope >= 0 ? "+" : ""}{temporal.trend_slope})</p>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-16 w-full">
        <polyline points={pts} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" />
        {s.map((v, i) => <circle key={i} cx={(i / (s.length - 1)) * W} cy={H - ((v - min) / (max - min + 1e-9)) * (H - 8) - 4} r="2.5" fill={color} />)}
      </svg>
    </div>
  );
}

/* ── Topographic head map ── */
export function HeadMap({ qeeg }: { qeeg: QEEG }) {
  const [band, setBand] = useState<keyof typeof BAND_COLORS>("alpha");
  // approximate 10-20 positions (unit circle)
  const POS: Record<string, [number, number]> = {
    Fp1: [-0.25, 0.85], Fp2: [0.25, 0.85], F7: [-0.7, 0.5], F3: [-0.35, 0.5], Fz: [0, 0.5], F4: [0.35, 0.5], F8: [0.7, 0.5],
    T3: [-0.9, 0], C3: [-0.45, 0], Cz: [0, 0], C4: [0.45, 0], T4: [0.9, 0],
    P3: [-0.35, -0.5], Pz: [0, -0.5], P4: [0.35, -0.5], O1: [-0.25, -0.85], O2: [0.25, -0.85],
  };
  const R = 90, cx = 100, cy = 100;
  return (
    <div className="panel p-5">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="flex items-center gap-2 text-sm font-semibold"><Brain className="h-4 w-4 text-[var(--color-cyan)]" /> Topographic map</h3>
        <div className="flex gap-1">
          {Object.keys(BAND_COLORS).map((b) => (
            <button key={b} onClick={() => setBand(b as keyof typeof BAND_COLORS)}
              className={cn("rounded px-2 py-0.5 text-[10px] capitalize", band === b ? "bg-white/10 text-[var(--color-ink)]" : "text-[var(--color-faint)]")}>{b}</button>
          ))}
        </div>
      </div>
      <svg viewBox="0 0 200 210" className="mx-auto h-56">
        <circle cx={cx} cy={cy} r={R} fill="none" stroke="var(--color-border)" strokeWidth="1.5" />
        <path d={`M${cx - 12},${cy - R} Q${cx},${cy - R - 14} ${cx + 12},${cy - R}`} fill="none" stroke="var(--color-border)" strokeWidth="1.5" />
        {qeeg.topography.map((ch) => {
          const pos = POS[ch.channel];
          if (!pos) return null;
          const x = cx + pos[0] * R, y = cy - pos[1] * R;
          const v = (ch.rel_band_power as unknown as Record<string, number>)[band] || 0;
          return (
            <g key={ch.channel}>
              <circle cx={x} cy={y} r={6 + v * 16} fill={BAND_COLORS[band]} opacity={0.25 + v} />
              <circle cx={x} cy={y} r={3} fill={BAND_COLORS[band]} />
              <text x={x} y={y + 14} textAnchor="middle" fontSize="6" className="fill-[var(--color-faint)]">{ch.channel}</text>
            </g>
          );
        })}
      </svg>
      <p className="text-center text-[10px] text-[var(--color-faint)]">relative {band} power per electrode (10-20 montage)</p>
    </div>
  );
}

/* ── qEEG parameters (expandable) ── */
export function QEEGParams({ qeeg }: { qeeg: QEEG }) {
  const [open, setOpen] = useState(false);
  const groups: Array<[string, Record<string, number | string>]> = [
    ["Band ratios", qeeg.spectral.band_ratios],
    ["Spectral", { sef95: qeeg.spectral.sef95, median_freq: qeeg.spectral.median_freq, peak_freq: qeeg.spectral.peak_freq, spectral_entropy: qeeg.spectral.spectral_entropy }],
    ["Time-domain", qeeg.time_domain],
    ["Complexity / entropy", qeeg.complexity],
    ["Connectivity", qeeg.connectivity as unknown as Record<string, number>],
    ["Quality", qeeg.quality],
  ];
  return (
    <div className="panel overflow-hidden">
      <button onClick={() => setOpen((o) => !o)} className="flex w-full items-center justify-between p-5">
        <h3 className="flex items-center gap-2 text-sm font-semibold"><Scan className="h-4 w-4 text-[var(--color-cyan)]" /> All qEEG parameters <span className="chip !py-0.5">{qeeg.n_features}</span></h3>
        <ChevronDown className={cn("h-4 w-4 transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <div className="grid gap-4 border-t border-[var(--color-border)] p-5 sm:grid-cols-2 lg:grid-cols-3">
          {groups.map(([title, obj]) => (
            <div key={title}>
              <p className="mb-1.5 text-xs font-semibold text-[var(--color-cyan)]">{title}</p>
              <dl className="space-y-1">
                {Object.entries(obj).map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-2 text-xs">
                    <dt className="text-[var(--color-faint)]">{k.replace(/_/g, " ")}</dt>
                    <dd className="font-mono text-[var(--color-muted)]">{typeof v === "number" ? v : String(v)}</dd>
                  </div>
                ))}
              </dl>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
