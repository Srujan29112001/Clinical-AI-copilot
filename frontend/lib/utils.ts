export function cn(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

export function pct(x: number): string {
  return `${Math.round(x * 100)}%`;
}

export const URGENCY_STYLES: Record<string, { text: string; bg: string; ring: string; label: string }> = {
  Emergent: { text: "text-red-300", bg: "bg-red-500/10", ring: "ring-red-500/40", label: "Emergent" },
  Urgent: { text: "text-amber-300", bg: "bg-amber-500/10", ring: "ring-amber-500/40", label: "Urgent" },
  Routine: { text: "text-emerald-300", bg: "bg-emerald-500/10", ring: "ring-emerald-500/40", label: "Routine" },
};

export const SEVERITY_BY_INTERACTION: Record<string, string> = {
  contraindicated: "text-red-300 bg-red-500/10 border-red-500/30",
  major: "text-rose-300 bg-rose-500/10 border-rose-500/30",
  moderate: "text-amber-300 bg-amber-500/10 border-amber-500/30",
  minor: "text-stone-300 bg-stone-500/10 border-stone-500/30",
};

export const GROUP_COLORS: Record<string, string> = {
  disease: "#ff3b4e", // red (primary)
  marker: "#f59e0b",  // amber
  concept: "#ff5d8f", // pink-red
  drug: "#34d399",    // green (kept distinct for legend readability)
};

export function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}
