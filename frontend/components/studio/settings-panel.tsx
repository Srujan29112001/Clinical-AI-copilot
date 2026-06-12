"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Settings2, Cpu, Cloud, ChevronDown, KeyRound, Search, Loader2 } from "lucide-react";
import type { LLMConfig, LocalModelsResult } from "@/lib/types";
import { detectLocalModels, isLive } from "@/lib/api";
import { PROVIDER_CATALOG } from "@/lib/providers";
import { cn } from "@/lib/utils";

const modelsFor = (id: string) => PROVIDER_CATALOG.find((p) => p.id === id)?.models || [];

const PROVIDERS = [
  { id: "mock", label: "Demo (no key)", kind: "demo", model: "deterministic" },
  { id: "ollama", label: "Ollama (local GPU)", kind: "local", model: "llama3.1:8b", base: "http://localhost:11434/v1" },
  { id: "vllm", label: "vLLM (local GPU)", kind: "local", model: "meta-llama/Llama-3.1-8B-Instruct", base: "http://localhost:8001/v1" },
  { id: "lmstudio", label: "LM Studio (local)", kind: "local", model: "local-model", base: "http://localhost:1234/v1" },
  { id: "anthropic", label: "Anthropic Claude", kind: "api", model: "claude-3-5-sonnet-latest" },
  { id: "openai", label: "OpenAI", kind: "api", model: "gpt-4o-mini" },
  { id: "groq", label: "Groq", kind: "api", model: "llama-3.3-70b-versatile" },
  { id: "deepseek", label: "DeepSeek", kind: "api", model: "deepseek-chat" },
  { id: "mistral", label: "Mistral", kind: "api", model: "mistral-large-latest" },
  { id: "gemini", label: "Google Gemini", kind: "api", model: "gemini-2.0-flash" },
  { id: "openrouter", label: "OpenRouter", kind: "api", model: "meta-llama/llama-3.3-70b-instruct" },
];

export function SettingsPanel({ config, onChange }: { config: LLMConfig; onChange: (c: LLMConfig) => void }) {
  const [open, setOpen] = useState(false);
  const [detecting, setDetecting] = useState(false);
  const [local, setLocal] = useState<LocalModelsResult | null>(null);
  const provider = PROVIDERS.find((p) => p.id === (config.provider || "mock")) || PROVIDERS[0];

  const detect = async () => {
    setDetecting(true);
    setLocal(null);
    const res = await detectLocalModels(config.base_url || provider.base || "http://localhost:11434/v1");
    setLocal(res);
    setDetecting(false);
  };

  const select = (id: string) => {
    const p = PROVIDERS.find((x) => x.id === id)!;
    onChange({
      ...config,
      provider: id === "mock" ? "" : id,
      model: p.model === "deterministic" ? "" : p.model,
      base_url: p.base || "",
      api_key: p.kind === "api" ? config.api_key : "",
    });
  };

  return (
    <div className="panel-soft overflow-hidden">
      <button onClick={() => setOpen((o) => !o)} className="flex w-full items-center justify-between p-4">
        <span className="flex items-center gap-2 text-sm font-semibold">
          <Settings2 className="h-4 w-4 text-[var(--color-cyan)]" /> Inference settings
        </span>
        <span className="flex items-center gap-2">
          <span className="chip !py-0.5">
            {provider.kind === "local" ? <Cpu className="h-3 w-3" /> : provider.kind === "api" ? <Cloud className="h-3 w-3" /> : null}
            {provider.label}
          </span>
          <ChevronDown className={cn("h-4 w-4 transition-transform", open && "rotate-180")} />
        </span>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div initial={{ height: 0 }} animate={{ height: "auto" }} exit={{ height: 0 }} className="overflow-hidden">
            <div className="space-y-4 border-t border-[var(--color-border)] p-4">
              <div>
                <label className="mb-1.5 block text-xs font-medium text-[var(--color-muted)]">Provider</label>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                  {PROVIDERS.map((p) => (
                    <button
                      key={p.id}
                      onClick={() => select(p.id)}
                      className={cn(
                        "rounded-lg border px-2.5 py-2 text-left text-xs transition-colors",
                        (config.provider || "mock") === p.id
                          ? "border-[var(--color-cyan)] bg-[var(--color-cyan)]/10 text-[var(--color-ink)]"
                          : "border-[var(--color-border)] text-[var(--color-muted)] hover:border-[var(--color-cyan)]/40",
                      )}
                    >
                      <span className="flex items-center gap-1.5">
                        {p.kind === "local" ? <Cpu className="h-3 w-3" /> : p.kind === "api" ? <Cloud className="h-3 w-3" /> : <span className="h-3 w-3" />}
                        {p.label}
                      </span>
                    </button>
                  ))}
                </div>
              </div>

              {provider.kind !== "demo" && (
                <div className="grid gap-3 sm:grid-cols-2">
                  <Field label="Model">
                    <input
                      list={`models-${provider.id}`}
                      value={config.model || ""}
                      onChange={(e) => onChange({ ...config, model: e.target.value })}
                      placeholder={provider.model}
                      className="input"
                    />
                    <datalist id={`models-${provider.id}`}>
                      {(local?.models?.length ? local.models.map((m) => m.id) : modelsFor(provider.id)).map((m) => (
                        <option key={m} value={m} />
                      ))}
                    </datalist>
                  </Field>
                  {provider.kind === "local" ? (
                    <Field label="Base URL">
                      <input
                        value={config.base_url || ""}
                        onChange={(e) => onChange({ ...config, base_url: e.target.value })}
                        placeholder={provider.base}
                        className="input"
                      />
                    </Field>
                  ) : (
                    <Field label="API Key">
                      <div className="flex items-center gap-2 rounded-lg border border-[var(--color-border)] px-3">
                        <KeyRound className="h-3.5 w-3.5 text-[var(--color-faint)]" />
                        <input
                          type="password"
                          value={config.api_key || ""}
                          onChange={(e) => onChange({ ...config, api_key: e.target.value })}
                          placeholder="sk-…"
                          className="w-full bg-transparent py-2 text-sm outline-none"
                        />
                      </div>
                    </Field>
                  )}
                </div>
              )}

              {provider.kind === "local" && (
                <div className="rounded-lg border border-[var(--color-border)] p-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium text-[var(--color-muted)]">Local GPU line-up</span>
                    <button onClick={detect} disabled={detecting}
                      className="btn btn-ghost !px-3 !py-1 !text-xs disabled:opacity-50">
                      {detecting ? <Loader2 className="h-3 w-3 animate-spin" /> : <Search className="h-3 w-3" />} Detect models
                    </button>
                  </div>
                  {local && (
                    <div className="mt-2 text-xs">
                      {local.reachable ? (
                        local.models.length ? (
                          <div className="flex flex-wrap gap-1.5">
                            {local.models.map((m) => (
                              <button key={m.id} onClick={() => onChange({ ...config, model: m.id })}
                                className={cn("rounded-md border px-2 py-1 transition-colors",
                                  config.model === m.id ? "border-[var(--color-cyan)] text-[var(--color-cyan)]" : "border-[var(--color-border)] text-[var(--color-muted)] hover:border-[var(--color-cyan)]/40")}>
                                {m.id}{m.size_gb ? ` · ${m.size_gb}GB` : ""}{m.quant ? ` · ${m.quant}` : ""}
                              </button>
                            ))}
                          </div>
                        ) : <span className="text-amber-300">Server reachable but no models installed.</span>
                      ) : (
                        <span className="text-[var(--color-faint)]">
                          {isLive() ? "No local server reachable at that URL — is Ollama/vLLM running?"
                            : "Live detection needs the backend running (NEXT_PUBLIC_API_URL). The list above shows common models."}
                        </span>
                      )}
                    </div>
                  )}
                </div>
              )}

              <p className="text-xs text-[var(--color-faint)]">
                {provider.kind === "local"
                  ? "Runs entirely on your machine — start the local server first. No data leaves your network."
                  : provider.kind === "api"
                  ? "Your key is sent only to the backend for this request and never stored."
                  : "Deterministic offline reasoning — works with zero configuration."}
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <style jsx>{`
        .input {
          width: 100%;
          border-radius: 0.5rem;
          border: 1px solid var(--color-border);
          background: var(--color-bg-soft);
          padding: 0.5rem 0.75rem;
          font-size: 0.875rem;
          outline: none;
        }
        .input:focus { border-color: var(--color-cyan); }
      `}</style>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="mb-1.5 block text-xs font-medium text-[var(--color-muted)]">{label}</label>
      {children}
    </div>
  );
}
