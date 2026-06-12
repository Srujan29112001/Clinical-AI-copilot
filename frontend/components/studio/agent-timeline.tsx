"use client";
import { motion, AnimatePresence } from "framer-motion";
import { Check, Loader2, Circle } from "lucide-react";
import { AGENTS } from "@/lib/agents";
import { AgentIcon } from "@/components/ui/agent-icon";
import type { AgentStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

export interface AgentState {
  status: AgentStatus;
  message?: string;
}

export function AgentTimeline({
  states, variant = "list",
}: {
  states: Record<string, AgentState>;
  variant?: "list" | "grid";
}) {
  if (variant === "grid") {
    return (
      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2 xl:grid-cols-3">
        {AGENTS.map((a) => {
          const st = states[a.id]?.status || "pending";
          const msg = states[a.id]?.message;
          const active = st === "running";
          const done = st === "done";
          return (
            <motion.div
              key={a.id}
              layout
              className={cn(
                "panel-soft min-w-0 p-3 transition-colors",
                active && "border-[var(--color-cyan)]/50 bg-[var(--color-cyan)]/[0.04]",
                done && "border-emerald-500/30",
              )}
            >
              <div className="flex items-center gap-2.5">
                <div className={cn(
                  "flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ring-1",
                  active ? "bg-[var(--color-cyan)]/15 ring-[var(--color-cyan)]/50"
                    : done ? "bg-emerald-500/15 ring-emerald-500/40"
                    : "bg-[var(--color-panel)] ring-[var(--color-border)]",
                )}>
                  <AgentIcon name={a.icon} className={cn("h-4 w-4", active ? "text-[var(--color-cyan)]" : done ? "text-emerald-400" : "text-[var(--color-faint)]")} />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-1.5">
                    <span className={cn("truncate text-xs font-semibold", st === "pending" && "text-[var(--color-faint)]")}>{a.name}</span>
                    <StatusBadge status={st} compact />
                  </div>
                </div>
              </div>
              <AnimatePresence>
                {msg && (active || done) && (
                  <motion.p
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    className="mt-2 line-clamp-2 break-words text-[11px] leading-relaxed text-[var(--color-muted)]"
                  >
                    {msg}
                  </motion.p>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {AGENTS.map((a, i) => {
        const st = states[a.id]?.status || "pending";
        const msg = states[a.id]?.message;
        const active = st === "running";
        const done = st === "done";
        return (
          <motion.div
            key={a.id}
            layout
            className={cn(
              "panel-soft flex gap-3 p-3.5 transition-colors",
              active && "border-[var(--color-cyan)]/50 bg-[var(--color-cyan)]/[0.04]",
              done && "border-emerald-500/30",
            )}
          >
            <div className="relative flex flex-col items-center">
              <div
                className={cn(
                  "flex h-9 w-9 items-center justify-center rounded-lg ring-1",
                  active ? "bg-[var(--color-cyan)]/15 ring-[var(--color-cyan)]/50"
                    : done ? "bg-emerald-500/15 ring-emerald-500/40"
                    : "bg-[var(--color-panel)] ring-[var(--color-border)]",
                )}
              >
                <AgentIcon name={a.icon} className={cn("h-4.5 w-4.5", active ? "text-[var(--color-cyan)]" : done ? "text-emerald-400" : "text-[var(--color-faint)]")} />
              </div>
              {i < AGENTS.length - 1 && (
                <div className={cn("mt-1 w-px flex-1", done ? "bg-emerald-500/40" : "bg-[var(--color-border)]")} style={{ minHeight: 8 }} />
              )}
            </div>

            <div className="min-w-0 flex-1">
              <div className="flex items-center justify-between gap-2">
                <span className={cn("text-sm font-semibold", st === "pending" && "text-[var(--color-faint)]")}>
                  {a.name}
                </span>
                <StatusBadge status={st} />
              </div>
              <p className="mt-0.5 text-xs text-[var(--color-faint)]">{a.role}</p>
              <AnimatePresence>
                {msg && (active || done) && (
                  <motion.p
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-2 line-clamp-3 break-words text-xs leading-relaxed text-[var(--color-muted)]"
                  >
                    {msg}
                  </motion.p>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}

function StatusBadge({ status, compact }: { status: AgentStatus; compact?: boolean }) {
  if (status === "running")
    return <span className="inline-flex shrink-0 items-center gap-1 text-[11px] text-[var(--color-cyan)]"><Loader2 className="h-3 w-3 animate-spin" />{!compact && " running"}</span>;
  if (status === "done")
    return <span className="inline-flex shrink-0 items-center gap-1 text-[11px] text-emerald-400"><Check className="h-3 w-3" />{!compact && " done"}</span>;
  return <span className="inline-flex shrink-0 items-center gap-1 text-[11px] text-[var(--color-faint)]"><Circle className="h-2.5 w-2.5" />{!compact && " queued"}</span>;
}
