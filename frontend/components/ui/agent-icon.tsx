import {
  Activity, Siren, Network, Stethoscope, Pill, ShieldCheck, FileText, Cpu,
  type LucideIcon,
} from "lucide-react";

const MAP: Record<string, LucideIcon> = {
  activity: Activity,
  siren: Siren,
  network: Network,
  stethoscope: Stethoscope,
  pill: Pill,
  shield: ShieldCheck,
  "file-text": FileText,
};

export function AgentIcon({ name, className }: { name: string; className?: string }) {
  const Icon = MAP[name] || Cpu;
  return <Icon className={className} />;
}
