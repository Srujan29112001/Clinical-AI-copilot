"use client";
import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, Cpu, Activity, Sparkles } from "lucide-react";
import { NeuralField } from "@/components/backgrounds/neural-field";

export function Hero() {
  return (
    <section className="relative overflow-hidden">
      <div className="absolute inset-0 -z-10">
        <NeuralField />
      </div>
      <div className="absolute inset-0 -z-10 grid-bg opacity-40" />

      <div className="container-page relative flex min-h-[calc(100vh-4rem)] flex-col items-center justify-center py-20 text-center">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="chip mb-6"
        >
          <Sparkles className="h-3.5 w-3.5 text-[var(--color-cyan)]" />
          Multi-agent clinical intelligence · v2.0
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.05 }}
          className="max-w-4xl text-balance text-4xl font-extrabold leading-[1.05] tracking-tight sm:text-6xl md:text-7xl"
        >
          From raw EEG to a <span className="gradient-text">clinical decision</span> — in seconds.
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.15 }}
          className="mt-6 max-w-2xl text-pretty text-lg text-[var(--color-muted)]"
        >
          Upload an EEG or clinical dataset and watch seven specialized AI agents triage,
          analyze the signal, retrieve evidence, diagnose, check drug safety and write the
          report — running on <strong className="text-[var(--color-ink)]">your local GPU</strong> or
          {" "}<strong className="text-[var(--color-ink)]">any API provider</strong>.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.25 }}
          className="mt-9 flex flex-wrap items-center justify-center gap-3"
        >
          <Link href="/studio" className="btn btn-primary text-base !px-6 !py-3">
            Open the Studio <ArrowRight className="h-4 w-4" />
          </Link>
          <Link href="/graph" className="btn btn-ghost text-base !px-6 !py-3">
            Explore the knowledge graph
          </Link>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mt-12 flex flex-wrap items-center justify-center gap-x-8 gap-y-3 text-sm text-[var(--color-faint)]"
        >
          <span className="inline-flex items-center gap-2"><Cpu className="h-4 w-4 text-[var(--color-teal)]" /> Hybrid local + cloud inference</span>
          <span className="inline-flex items-center gap-2"><Activity className="h-4 w-4 text-[var(--color-violet)]" /> Real-time agent streaming</span>
          <span className="inline-flex items-center gap-2"><Sparkles className="h-4 w-4 text-[var(--color-cyan)]" /> Zero-key live demo</span>
        </motion.div>
      </div>
    </section>
  );
}
