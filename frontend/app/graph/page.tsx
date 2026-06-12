"use client";
import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Network, X, Maximize2, Info } from "lucide-react";
import { ForceGraph } from "@/components/graph/force-graph";
import { fetchKnowledgeGraph } from "@/lib/api";
import type { GraphNode, KnowledgeGraph } from "@/lib/types";
import { GROUP_COLORS, cn } from "@/lib/utils";

const GROUP_LABELS: Record<string, string> = {
  disease: "Diseases (ICD-10)",
  marker: "EEG markers",
  concept: "Concepts (SNOMED-CT)",
  drug: "Drugs (RxNorm)",
};

export default function GraphPage() {
  const [graph, setGraph] = useState<KnowledgeGraph | null>(null);
  const [active, setActive] = useState<Set<string>>(new Set());
  const [selected, setSelected] = useState<GraphNode | null>(null);

  useEffect(() => {
    fetchKnowledgeGraph().then((g) => {
      setGraph(g);
      setActive(new Set(g.groups));
    });
  }, []);

  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    graph?.nodes.forEach((n) => { c[n.group] = (c[n.group] || 0) + 1; });
    return c;
  }, [graph]);

  const toggle = (g: string) => {
    setActive((prev) => {
      const next = new Set(prev);
      next.has(g) ? next.delete(g) : next.add(g);
      return next;
    });
  };

  return (
    <div className="container-page py-10">
      <header className="mb-6">
        <h1 className="flex items-center gap-2 text-3xl font-bold tracking-tight">
          <Network className="h-7 w-7 text-[var(--color-cyan)]" /> Clinical Knowledge Graph
        </h1>
        <p className="mt-2 max-w-2xl text-[var(--color-muted)]">
          The GraphRAG layer that grounds every diagnosis — ICD-10 diseases, EEG markers,
          SNOMED-CT concepts and RxNorm drugs, with their clinical relationships. Drag nodes,
          scroll to zoom, click for detail.
        </p>
      </header>

      <div className="grid gap-5 lg:grid-cols-[1fr_300px]">
        {/* graph canvas */}
        <div className="panel relative h-[600px] overflow-hidden">
          {graph ? (
            <>
              <ForceGraph graph={graph} activeGroups={active} onSelect={setSelected} />
              <div className="pointer-events-none absolute left-4 top-4 chip">
                <Maximize2 className="h-3 w-3" /> scroll to zoom · drag to pan
              </div>
            </>
          ) : (
            <div className="flex h-full items-center justify-center text-[var(--color-muted)]">
              <div className="h-6 w-6 animate-spin rounded-full border-2 border-[var(--color-cyan)] border-t-transparent" />
            </div>
          )}

          {/* detail card */}
          {selected && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="absolute bottom-4 right-4 w-72 panel-soft p-4"
            >
              <button onClick={() => setSelected(null)} className="absolute right-3 top-3 rounded p-0.5 hover:bg-white/10">
                <X className="h-4 w-4" />
              </button>
              <span className="chip !py-0.5" style={{ color: GROUP_COLORS[selected.group] }}>
                {GROUP_LABELS[selected.group] || selected.group}
              </span>
              <h3 className="mt-2 font-semibold">{selected.label}</h3>
              {selected.title && <p className="mt-1 text-xs text-[var(--color-muted)]">{selected.title}</p>}
              <dl className="mt-3 space-y-1 text-xs">
                {selected.category && <Row k="Category" v={selected.category} />}
                {selected.severity && <Row k="Severity" v={selected.severity} />}
                {selected.semantic && <Row k="Type" v={selected.semantic} />}
                {selected.drug_class && <Row k="Class" v={selected.drug_class} />}
                {selected.dose && <Row k="Dose" v={selected.dose} />}
                {selected.brands?.length ? <Row k="Brands" v={selected.brands.join(", ")} /> : null}
              </dl>
            </motion.div>
          )}
        </div>

        {/* legend / filters */}
        <div className="space-y-4">
          <div className="panel p-5">
            <h3 className="mb-3 text-sm font-semibold">Node types</h3>
            <div className="space-y-2">
              {graph?.groups.map((g) => (
                <button
                  key={g}
                  onClick={() => toggle(g)}
                  className={cn(
                    "flex w-full items-center justify-between rounded-lg border px-3 py-2 text-sm transition-colors",
                    active.has(g) ? "border-[var(--color-border)]" : "border-transparent opacity-40",
                  )}
                >
                  <span className="flex items-center gap-2">
                    <span className="h-3 w-3 rounded-full" style={{ background: GROUP_COLORS[g] }} />
                    {GROUP_LABELS[g] || g}
                  </span>
                  <span className="font-mono text-xs text-[var(--color-faint)]">{counts[g] || 0}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="panel p-5">
            <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold"><Info className="h-4 w-4 text-[var(--color-cyan)]" /> About</h3>
            <p className="text-xs leading-relaxed text-[var(--color-muted)]">
              Built from the project&apos;s ICD-10, SNOMED-CT and RxNorm ontologies. The
              Knowledge Retriever agent traverses these relationships (disease → EEG marker,
              disease → treatment) to ground each diagnosis in structured clinical evidence —
              a portable in-memory stand-in for the production Neo4j + Qdrant GraphRAG.
            </p>
            {graph && (
              <div className="mt-3 flex gap-4 text-xs">
                <span><strong className="text-[var(--color-ink)]">{graph.nodes.length}</strong> nodes</span>
                <span><strong className="text-[var(--color-ink)]">{graph.edges.length}</strong> edges</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-2">
      <dt className="text-[var(--color-faint)]">{k}</dt>
      <dd className="text-right capitalize text-[var(--color-muted)]">{v}</dd>
    </div>
  );
}
