"use client";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { Send, Bot, User, Sparkles, FileText, Loader2 } from "lucide-react";
import { streamChat, isLive } from "@/lib/api";
import { useLLMConfig, useLastResult, useLastPatient } from "@/lib/store";
import { cn } from "@/lib/utils";

interface Msg { role: "user" | "assistant"; content: string }

const SUGGESTIONS = [
  "Is this patient at risk of a seizure?",
  "Explain the EEG findings in plain language.",
  "Are there any drug interactions I should worry about?",
  "What's the recommended treatment and why?",
];

export default function ChatPage() {
  const [llm] = useLLMConfig();
  const [result] = useLastResult();
  const [patient] = useLastPatient();
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, streaming]);

  const send = async (text: string) => {
    const q = text.trim();
    if (!q || streaming) return;
    const next: Msg[] = [...messages, { role: "user", content: q }];
    setMessages(next);
    setInput("");
    setStreaming(true);
    setMessages((m) => [...m, { role: "assistant", content: "" }]);

    await streamChat(
      next.map((m) => ({ role: m.role, content: m.content })),
      { patient: patient || undefined, analysis: result },
      llm,
      {
        onToken: (t) =>
          setMessages((m) => {
            const copy = [...m];
            copy[copy.length - 1] = { role: "assistant", content: copy[copy.length - 1].content + t };
            return copy;
          }),
        onDone: () => setStreaming(false),
        onError: () => setStreaming(false),
      },
    );
    setStreaming(false);
  };

  return (
    <div className="container-page py-10">
      <div className="mx-auto max-w-3xl">
        <header className="mb-6">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-3xl font-bold tracking-tight">Clinical AI Copilot — Chat</h1>
            <span className="chip">
              <span className={`h-1.5 w-1.5 rounded-full ${isLive() ? "bg-emerald-400" : "bg-amber-400"} animate-pulse-glow`} />
              {isLive() ? "Live model" : "Offline demo"}
            </span>
          </div>
          <p className="mt-2 text-[var(--color-muted)]">
            Ask about the latest analysis — EEG findings, diagnosis, drug safety, treatment.
            {result ? null : " Run an analysis in the Studio first to ground the conversation."}
          </p>
        </header>

        {/* context banner */}
        {result && (
          <Link href="/studio" className="panel-soft mb-4 flex items-center gap-3 p-3 text-sm hover:border-[var(--color-cyan)]/40">
            <FileText className="h-4 w-4 text-[var(--color-cyan)]" />
            <span className="text-[var(--color-muted)]">Context loaded:</span>
            <span className="font-medium">{result.primary_diagnosis.condition.slice(0, 48)}</span>
            <span className="chip ml-auto !py-0.5">{result.urgency}</span>
          </Link>
        )}

        {/* messages */}
        <div className="panel min-h-[420px] p-5">
          {messages.length === 0 ? (
            <div className="flex h-[380px] flex-col items-center justify-center text-center">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-[var(--color-cyan)]/20 to-[var(--color-violet)]/20 ring-1 ring-[var(--color-border)]">
                <Bot className="h-7 w-7 text-[var(--color-cyan)]" />
              </div>
              <h3 className="mt-4 font-semibold">How can I help with this case?</h3>
              <div className="mt-5 grid w-full max-w-lg gap-2 sm:grid-cols-2">
                {SUGGESTIONS.map((s) => (
                  <button key={s} onClick={() => send(s)} className="panel-soft p-3 text-left text-sm text-[var(--color-muted)] transition-colors hover:border-[var(--color-cyan)]/40 hover:text-[var(--color-ink)]">
                    <Sparkles className="mb-1 h-3.5 w-3.5 text-[var(--color-cyan)]" />
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-5">
              {messages.map((m, i) => (
                <motion.div key={i} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className={cn("flex gap-3", m.role === "user" && "flex-row-reverse")}>
                  <div className={cn("flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ring-1", m.role === "user" ? "bg-[var(--color-violet)]/15 ring-[var(--color-violet)]/40" : "bg-[var(--color-cyan)]/15 ring-[var(--color-cyan)]/40")}>
                    {m.role === "user" ? <User className="h-4 w-4 text-[var(--color-violet)]" /> : <Bot className="h-4 w-4 text-[var(--color-cyan)]" />}
                  </div>
                  <div className={cn("max-w-[80%] min-w-0 whitespace-pre-wrap break-words rounded-2xl px-4 py-2.5 text-sm leading-relaxed", m.role === "user" ? "bg-[var(--color-violet)]/10 text-[var(--color-ink)]" : "bg-[var(--color-panel-2)] text-[var(--color-muted)]")}>
                    {m.content || (streaming && i === messages.length - 1 ? <Loader2 className="h-4 w-4 animate-spin" /> : null)}
                  </div>
                </motion.div>
              ))}
              <div ref={endRef} />
            </div>
          )}
        </div>

        {/* input */}
        <form
          onSubmit={(e) => { e.preventDefault(); send(input); }}
          className="mt-4 flex items-center gap-2 panel-soft p-2"
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about the EEG, diagnosis, drug safety…"
            className="flex-1 bg-transparent px-3 py-2 text-sm outline-none"
          />
          <button type="submit" disabled={streaming || !input.trim()} className="btn btn-primary !px-4 !py-2 disabled:opacity-50">
            {streaming ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </button>
        </form>
        <p className="mt-2 text-center text-xs text-[var(--color-faint)]">
          Decision-support only · Configure a local or API model in the Studio for full reasoning.
        </p>
      </div>
    </div>
  );
}
