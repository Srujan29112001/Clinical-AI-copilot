"use client";
import { useEffect, useRef } from "react";

/**
 * Animated neural field — drifting neon nodes connected by synapse lines, with
 * an EEG-style traveling pulse. Pure canvas, disables on mobile / reduced-motion.
 */
export function NeuralField({ density = 64 }: { density?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const mobile = window.innerWidth < 640;
    const count = mobile ? Math.round(density * 0.45) : density;

    let w = 0, h = 0, raf = 0, t = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    type Node = { x: number; y: number; vx: number; vy: number; r: number };
    const nodes: Node[] = [];

    const resize = () => {
      const parent = canvas.parentElement!;
      w = parent.clientWidth;
      h = parent.clientHeight;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = w + "px";
      canvas.style.height = h + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();

    for (let i = 0; i < count; i++) {
      nodes.push({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.25,
        vy: (Math.random() - 0.5) * 0.25,
        r: 1 + Math.random() * 2,
      });
    }

    const LINK = mobile ? 110 : 150;

    const draw = () => {
      t += 1;
      ctx.clearRect(0, 0, w, h);

      // links
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j];
          const dx = a.x - b.x, dy = a.y - b.y;
          const d = Math.hypot(dx, dy);
          if (d < LINK) {
            const alpha = (1 - d / LINK) * 0.5;
            const grad = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
            grad.addColorStop(0, `rgba(34,211,238,${alpha})`);
            grad.addColorStop(1, `rgba(167,139,250,${alpha})`);
            ctx.strokeStyle = grad;
            ctx.lineWidth = 0.6;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }

      // nodes
      for (const n of nodes) {
        n.x += n.vx;
        n.y += n.vy;
        if (n.x < 0 || n.x > w) n.vx *= -1;
        if (n.y < 0 || n.y > h) n.vy *= -1;
        const pulse = 0.6 + 0.4 * Math.sin((t + n.x) * 0.02);
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(45,212,191,${0.5 * pulse})`;
        ctx.shadowBlur = 8;
        ctx.shadowColor = "rgba(34,211,238,0.6)";
        ctx.fill();
        ctx.shadowBlur = 0;
      }

      // traveling EEG pulse line
      const baseY = h * 0.5;
      ctx.beginPath();
      for (let x = 0; x <= w; x += 4) {
        const phase = (x + t * 2) * 0.02;
        const spike = Math.exp(-(((x - ((t * 2) % (w + 200))) / 18) ** 2)) * 40;
        const y = baseY + Math.sin(phase) * 6 + Math.sin(phase * 3) * 3 - spike;
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = "rgba(34,211,238,0.18)";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      raf = requestAnimationFrame(draw);
    };

    if (reduce) {
      draw(); // single frame
    } else {
      draw();
    }

    window.addEventListener("resize", resize);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, [density]);

  return <canvas ref={ref} className="absolute inset-0 h-full w-full" aria-hidden />;
}
