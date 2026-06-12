import type { ProviderInfo } from "./types";

// Static mirror of backend/app/llm.py PROVIDERS+MODELS (offline fallback).
export const PROVIDER_CATALOG: ProviderInfo[] = [
  { id: "mock", default_model: "deterministic", models: ["deterministic"], default_base_url: "", local: false, needs_key: false, kind: "demo" },
  { id: "ollama", default_model: "llama3.1:8b", models: ["llama3.1:8b", "llama3.2:3b", "mistral:7b", "qwen2.5:7b", "phi3:mini", "gemma2:9b"], default_base_url: "http://localhost:11434/v1", local: true, needs_key: false, kind: "openai-compatible" },
  { id: "vllm", default_model: "meta-llama/Llama-3.1-8B-Instruct", models: ["meta-llama/Llama-3.1-8B-Instruct"], default_base_url: "http://localhost:8001/v1", local: true, needs_key: false, kind: "openai-compatible" },
  { id: "lmstudio", default_model: "local-model", models: ["local-model"], default_base_url: "http://localhost:1234/v1", local: true, needs_key: false, kind: "openai-compatible" },
  { id: "anthropic", default_model: "claude-sonnet-4-6", models: ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-fable-5"], default_base_url: "https://api.anthropic.com/v1", local: false, needs_key: true, kind: "anthropic" },
  { id: "openai", default_model: "gpt-5-4-mini", models: ["gpt-5-5", "gpt-5-4", "gpt-5-4-mini", "gpt-5-4-nano", "gpt-4-1", "o3", "o4-mini"], default_base_url: "https://api.openai.com/v1", local: false, needs_key: true, kind: "openai-compatible" },
  { id: "groq", default_model: "llama-3.3-70b-versatile", models: ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"], default_base_url: "https://api.groq.com/openai/v1", local: false, needs_key: true, kind: "openai-compatible" },
  { id: "deepseek", default_model: "deepseek-v4-flash", models: ["deepseek-v4-flash", "deepseek-v4-pro"], default_base_url: "https://api.deepseek.com/v1", local: false, needs_key: true, kind: "openai-compatible" },
  { id: "mistral", default_model: "mistral-large-2512", models: ["mistral-large-2512", "mistral-small-latest", "ministral-8b-latest"], default_base_url: "https://api.mistral.ai/v1", local: false, needs_key: true, kind: "openai-compatible" },
  { id: "gemini", default_model: "gemini-3.5-flash", models: ["gemini-3.5-flash", "gemini-3.5-pro", "gemini-2.5-flash"], default_base_url: "https://generativelanguage.googleapis.com/v1beta/openai", local: false, needs_key: true, kind: "openai-compatible" },
  { id: "openrouter", default_model: "anthropic/claude-sonnet-4-6", models: ["anthropic/claude-sonnet-4-6", "openai/gpt-5-4", "qwen/qwen3-coder:free", "moonshot/kimi-k2-6:free", "meta-llama/llama-3.3-70b-instruct"], default_base_url: "https://openrouter.ai/api/v1", local: false, needs_key: true, kind: "openai-compatible" },
];
