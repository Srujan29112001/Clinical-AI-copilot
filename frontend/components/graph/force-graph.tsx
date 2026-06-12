"use client";
import { useEffect, useRef, useState } from "react";
import type { GraphNode, KnowledgeGraph } from "@/lib/types";
import { GROUP_COLORS } from "@/lib/utils";

interface Sim extends GraphNode { x: number; y: number; vx: number; vy: number; deg: number }

export function ForceGraph({
  graph, activeGroups, onSelect,
}: {
  graph: KnowledgeGraph;
  activeGroups: Set<string>;
  onSelect: (n: GraphNode | null) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<string | null>(null);
  const stateRef = useRef<{ nodes: Sim[]; edges: { a: Sim; b: Sim; label: string }[] }>({ nodes: [], edges: [] });
  const viewRef = useRef({ scale: 1, ox: 0, oy: 0 });
  const dragRef = useRef<{ node: Sim | null; panning: boolean; lx: number; ly: number }>({ node: null, panning: false, lx: 0, ly: 0 });

  // build sim when graph changes
  useEffect(() => {
    const idMap = new Map<string, Sim>();
    const nodes: Sim[] = graph.nodes.map((n, i) => {
      const angle = (i / graph.nodes.length) * Math.PI * 2;
      const s: Sim = { ...n, x: Math.cos(angle) * 220 + (Math.random() - 0.5) * 60, y: Math.sin(angle) * 220 + (Math.random() - 0.5) * 60, vx: 0, vy: 0, deg: 0 };
      idMap.set(n.id, s);
      return s;
    });
    const edges = graph.edges
      .map((e) => ({ a: idMap.get(e.source)!, b: idMap.get(e.target)!, label: e.label }))
      .filter((e) => e.a && e.b);
    edges.forEach((e) => { e.a.deg++; e.b.deg++; });
    stateRef.current = { nodes, edges };
  }, [graph]);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let raf = 0, frame = 0;

    const resize = () => {
      const p = canvas.parentElement!;
      canvas.width = p.clientWidth * dpr;
      canvas.height = p.clientHeight * dpr;
      canvas.style.width = p.clientWidth + "px";
      canvas.style.height = p.clientHeight + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      viewRef.current.ox = p.clientWidth / 2;
      viewRef.current.oy = p.clientHeight / 2;
    };
    resize();

    const step = () => {
      const { nodes, edges } = stateRef.current;
      const visible = (n: Sim) => activeGroups.has(n.group);
      frame++;
      const cool = frame < 240 ? 1 : 0.15; // settle then idle

      // repulsion
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        if (!visible(a)) continue;
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j];
          if (!visible(b)) continue;
          let dx = a.x - b.x, dy = a.y - b.y;
          let d2 = dx * dx + dy * dy || 0.01;
          const f = (2600 / d2) * cool;
          const d = Math.sqrt(d2);
          dx /= d; dy /= d;
          a.vx += dx * f; a.vy += dy * f;
          b.vx -= dx * f; b.vy -= dy * f;
        }
      }
      // springs
      for (const e of edges) {
        if (!visible(e.a) || !visible(e.b)) continue;
        let dx = e.b.x - e.a.x, dy = e.b.y - e.a.y;
        const d = Math.hypot(dx, dy) || 0.01;
        const f = (d - 90) * 0.015 * cool;
        dx /= d; dy /= d;
        e.a.vx += dx * f; e.a.vy += dy * f;
        e.b.vx -= dx * f; e.b.vy -= dy * f;
      }
      // centering + integrate
      for (const n of nodes) {
        if (!visible(n)) continue;
        n.vx += -n.x * 0.0016 * cool;
        n.vy += -n.y * 0.0016 * cool;
        n.vx *= 0.86; n.vy *= 0.86;
        if (dragRef.current.node !== n) { n.x += n.vx; n.y += n.vy; }
      }
    };

    const draw = () => {
      const { nodes, edges } = stateRef.current;
      const { scale, ox, oy } = viewRef.current;
      const visible = (n: Sim) => activeGroups.has(n.group);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.translate(ox, oy);
      ctx.scale(scale, scale);

      // edges
      for (const e of edges) {
        if (!visible(e.a) || !visible(e.b)) continue;
        const hot = hover === e.a.id || hover === e.b.id;
        ctx.strokeStyle = hot ? "rgba(255,59,78,0.55)" : "rgba(180,130,140,0.16)";
        ctx.lineWidth = hot ? 1.4 : 0.7;
        ctx.beginPath();
        ctx.moveTo(e.a.x, e.a.y);
        ctx.lineTo(e.b.x, e.b.y);
        ctx.stroke();
      }
      // nodes
      for (const n of nodes) {
        if (!visible(n)) continue;
        const r = 4 + Math.min(n.deg, 10) * 1.1;
        const color = GROUP_COLORS[n.group] || "#8ea0c4";
        const hot = hover === n.id;
        ctx.beginPath();
        ctx.arc(n.x, n.y, hot ? r + 2 : r, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.globalAlpha = hot ? 1 : 0.9;
        ctx.shadowBlur = hot ? 16 : 6;
        ctx.shadowColor = color;
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.globalAlpha = 1;
        if (hot || scale > 1.3 || n.deg > 4) {
          ctx.fillStyle = "#eaf0ff";
          ctx.font = `${hot ? 12 : 10}px Inter, sans-serif`;
          ctx.fillText(n.label.slice(0, 22), n.x + r + 3, n.y + 3);
        }
      }
      ctx.restore();
    };

    const loop = () => { step(); draw(); raf = requestAnimationFrame(loop); };
    loop();

    // ── interaction ──
    const toWorld = (cx: number, cy: number) => {
      const { scale, ox, oy } = viewRef.current;
      return { x: (cx - ox) / scale, y: (cy - oy) / scale };
    };
    const pick = (cx: number, cy: number): Sim | null => {
      const w = toWorld(cx, cy);
      let best: Sim | null = null, bd = 16;
      for (const n of stateRef.current.nodes) {
        if (!activeGroups.has(n.group)) continue;
        const d = Math.hypot(n.x - w.x, n.y - w.y);
        if (d < bd) { bd = d; best = n; }
      }
      return best;
    };
    const rect = () => canvas.getBoundingClientRect();
    const onMove = (ev: MouseEvent) => {
      const r = rect();
      const cx = ev.clientX - r.left, cy = ev.clientY - r.top;
      if (dragRef.current.node) {
        const w = toWorld(cx, cy);
        dragRef.current.node.x = w.x; dragRef.current.node.y = w.y;
        dragRef.current.node.vx = dragRef.current.node.vy = 0;
        return;
      }
      if (dragRef.current.panning) {
        viewRef.current.ox += cx - dragRef.current.lx;
        viewRef.current.oy += cy - dragRef.current.ly;
        dragRef.current.lx = cx; dragRef.current.ly = cy;
        return;
      }
      const n = pick(cx, cy);
      setHover(n?.id || null);
      canvas.style.cursor = n ? "pointer" : "grab";
    };
    const onDown = (ev: MouseEvent) => {
      const r = rect();
      const cx = ev.clientX - r.left, cy = ev.clientY - r.top;
      const n = pick(cx, cy);
      if (n) { dragRef.current.node = n; onSelect(n); }
      else { dragRef.current.panning = true; dragRef.current.lx = cx; dragRef.current.ly = cy; }
    };
    const onUp = () => { dragRef.current.node = null; dragRef.current.panning = false; };
    const onWheel = (ev: WheelEvent) => {
      ev.preventDefault();
      const factor = ev.deltaY < 0 ? 1.1 : 0.9;
      viewRef.current.scale = Math.max(0.3, Math.min(3, viewRef.current.scale * factor));
    };

    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mousedown", onDown);
    window.addEventListener("mouseup", onUp);
    canvas.addEventListener("wheel", onWheel, { passive: false });
    window.addEventListener("resize", resize);
    return () => {
      cancelAnimationFrame(raf);
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mousedown", onDown);
      window.removeEventListener("mouseup", onUp);
      canvas.removeEventListener("wheel", onWheel);
      window.removeEventListener("resize", resize);
    };
  }, [activeGroups, hover, onSelect]);

  return <canvas ref={canvasRef} className="h-full w-full" />;
}
