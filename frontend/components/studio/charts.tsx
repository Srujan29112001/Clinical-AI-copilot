"use client";
import type { BandPowers } from "@/lib/types";

const BAND_COLORS: Record<string, string> = {
  delta: "#6366f1", theta: "#22d3ee", alpha: "#2dd4bf", beta: "#f59e0b", gamma: "#fb7185",
};

export function WaveformChart({ data }: { data: Array<{ t: number; v: number }> }) {
  if (!data?.length) return null;
  const W = 520, H = 120, mid = H / 2;
  const step = W / (data.length - 1);
  const path = data
    .map((d, i) => `${i === 0 ? "M" : "L"} ${(i * step).toFixed(1)} ${(mid - d.v * (mid - 8)).toFixed(1)}`)
    .join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-28 w-full" preserveAspectRatio="none">
      <line x1="0" y1={mid} x2={W} y2={mid} stroke="var(--color-border)" strokeWidth="1" />
      <path d={path} fill="none" stroke="var(--color-cyan)" strokeWidth="1.6" strokeLinejoin="round" />
    </svg>
  );
}

export function PSDChart({ data }: { data: Array<{ freq: number; power: number }> }) {
  if (!data?.length) return null;
  const W = 520, H = 130, pad = 4;
  const step = W / data.length;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-32 w-full" preserveAspectRatio="none">
      {data.map((d, i) => {
        const h = d.power * (H - pad * 2);
        // colour bars by EEG band
        const f = d.freq;
        const c = f < 4 ? BAND_COLORS.delta : f < 8 ? BAND_COLORS.theta : f < 13 ? BAND_COLORS.alpha : f < 30 ? BAND_COLORS.beta : BAND_COLORS.gamma;
        return (
          <rect
            key={i}
            x={i * step + 0.5}
            y={H - pad - h}
            width={Math.max(step - 1, 1)}
            height={h}
            fill={c}
            opacity={0.85}
            rx={1}
          />
        );
      })}
    </svg>
  );
}

export function BandPowerBars({ bands }: { bands: BandPowers }) {
  const entries = Object.entries(bands) as Array<[string, number]>;
  const max = Math.max(...entries.map(([, v]) => v)) || 1;
  return (
    <div className="space-y-2.5">
      {entries.map(([name, v]) => (
        <div key={name} className="flex items-center gap-3">
          <span className="w-14 text-xs capitalize text-[var(--color-muted)]">{name}</span>
          <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-[var(--color-bg-soft)]">
            <div
              className="h-full rounded-full transition-all duration-700"
              style={{ width: `${(v / max) * 100}%`, background: BAND_COLORS[name] }}
            />
          </div>
          <span className="w-12 text-right font-mono text-xs text-[var(--color-ink)]">{(v * 100).toFixed(0)}%</span>
        </div>
      ))}
    </div>
  );
}

export function Donut({ value, label, color = "var(--color-cyan)" }: { value: number; label: string; color?: string }) {
  const r = 38, c = 2 * Math.PI * r;
  const off = c * (1 - value);
  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 100 100" className="h-28 w-28 -rotate-90">
        <circle cx="50" cy="50" r={r} fill="none" stroke="var(--color-border)" strokeWidth="9" />
        <circle
          cx="50" cy="50" r={r} fill="none" stroke={color} strokeWidth="9" strokeLinecap="round"
          strokeDasharray={c} strokeDashoffset={off}
          style={{ transition: "stroke-dashoffset 1s ease" }}
        />
        <text x="50" y="50" transform="rotate(90 50 50)" textAnchor="middle" dominantBaseline="central"
          className="fill-[var(--color-ink)] font-bold" fontSize="20">
          {Math.round(value * 100)}%
        </text>
      </svg>
      <span className="mt-1 text-xs text-[var(--color-muted)]">{label}</span>
    </div>
  );
}
