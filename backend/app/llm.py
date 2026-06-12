"""
Hybrid LLM layer — run clinical reasoning on a LOCAL GPU model *or* a hosted
API provider, selectable per request from the UI or via environment variables.

Design (inspired by the Helix project's provider abstraction):

  • LLMProvider          abstract base
  • MockLLM              deterministic, zero-key — powers the public demo
  • OpenAICompatibleLLM  Groq / OpenAI / DeepSeek / Mistral / OpenRouter /
                         Gemini (OpenAI-compat) / **Ollama** / **vLLM** (local GPU)
  • AnthropicLLM         native Claude Messages API

Every clinical agent ("role") resolves its own provider independently, so you
can, e.g., run the heavy Diagnostician on a hosted Claude model while the
Triage agent runs on a local Ollama model on your own GPU.

Resolution order for a role R (UPPER-cased):
    1. per-request override (LLMConfig sent from the UI)
    2. CLINICAL_<R>_API_KEY / _MODEL / _BASE_URL / _PROVIDER
    3. CLINICAL_LLM_API_KEY / _MODEL / _BASE_URL / _PROVIDER  (global default)
    4. MockLLM  (no key anywhere → demo mode)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import httpx

# provider id -> (default base_url, default model, is_anthropic)
PROVIDERS: dict[str, tuple[str, str, bool]] = {
    "anthropic":  ("https://api.anthropic.com/v1", "claude-sonnet-4-6", True),
    "openai":     ("https://api.openai.com/v1", "gpt-5-4-mini", False),
    "groq":       ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", False),
    "deepseek":   ("https://api.deepseek.com/v1", "deepseek-v4-flash", False),
    "mistral":    ("https://api.mistral.ai/v1", "mistral-large-2512", False),
    "openrouter": ("https://openrouter.ai/api/v1", "anthropic/claude-sonnet-4-6", False),
    "gemini":     ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.5-flash", False),
    # ── local GPU options (OpenAI-compatible servers) ──
    "ollama":     ("http://localhost:11434/v1", "llama3.1:8b", False),
    "vllm":       ("http://localhost:8001/v1", "meta-llama/Llama-3.1-8B-Instruct", False),
    "local":      ("http://localhost:11434/v1", "llama3.1:8b", False),
    "lmstudio":   ("http://localhost:1234/v1", "local-model", False),
}

# provider id -> curated current (2026) model ids for the UI picker
MODELS: dict[str, list[str]] = {
    "anthropic":  ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-fable-5"],
    "openai":     ["gpt-5-5", "gpt-5-4", "gpt-5-4-mini", "gpt-5-4-nano", "gpt-4-1", "o3", "o4-mini"],
    "groq":       ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    "deepseek":   ["deepseek-v4-flash", "deepseek-v4-pro"],
    "mistral":    ["mistral-large-2512", "mistral-small-latest", "ministral-8b-latest"],
    "openrouter": ["anthropic/claude-sonnet-4-6", "openai/gpt-5-4", "qwen/qwen3-coder:free",
                   "moonshot/kimi-k2-6:free", "meta-llama/llama-3.3-70b-instruct"],
    "gemini":     ["gemini-3.5-flash", "gemini-3.5-pro", "gemini-2.5-flash"],
    "ollama":     ["llama3.1:8b", "llama3.2:3b", "mistral:7b", "qwen2.5:7b", "phi3:mini", "gemma2:9b"],
    "vllm":       ["meta-llama/Llama-3.1-8B-Instruct"],
    "local":      ["llama3.1:8b"],
    "lmstudio":   ["local-model"],
}

LOCAL_PROVIDERS = {"ollama", "vllm", "local", "lmstudio"}


@dataclass
class Resolved:
    provider: str
    model: str
    api_key: Optional[str]
    base_url: str
    temperature: float
    max_tokens: int
    is_anthropic: bool

    @property
    def is_local(self) -> bool:
        return self.provider in LOCAL_PROVIDERS

    @property
    def runs_real(self) -> bool:
        """True if it will call a real model (local needs no key)."""
        return self.is_local or bool(self.api_key)


def _env(role: str, key: str) -> Optional[str]:
    return os.getenv(f"CLINICAL_{role.upper()}_{key}") or os.getenv(f"CLINICAL_LLM_{key}")


def resolve(role: str, override: Optional[dict] = None) -> Resolved:
    """Resolve the effective provider config for an agent role."""
    override = override or {}
    provider = (override.get("provider") or _env(role, "PROVIDER") or "").strip().lower()

    api_key = override.get("api_key") or _env(role, "API_KEY")
    base_url = override.get("base_url") or _env(role, "BASE_URL")
    model = override.get("model") or _env(role, "MODEL")

    # Infer provider when only a key is supplied (default to mock-free behaviour)
    if not provider:
        if base_url:
            provider = "local" if "localhost" in base_url or "127.0.0.1" in base_url else "openai"
        elif api_key:
            provider = "anthropic" if api_key.startswith("sk-ant") else "openai"
        else:
            provider = "mock"

    default_base, default_model, is_anthropic = PROVIDERS.get(
        provider, ("", "", False)
    )
    return Resolved(
        provider=provider,
        model=model or default_model,
        api_key=api_key,
        base_url=(base_url or default_base).rstrip("/"),
        temperature=float(override.get("temperature", 0.2)),
        max_tokens=int(override.get("max_tokens", 1024)),
        is_anthropic=is_anthropic,
    )


# ──────────────────────────────────────────────────────────────────────────
class LLMProvider:
    async def complete(self, system: str, prompt: str) -> str:  # pragma: no cover
        raise NotImplementedError

    async def stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        # Default: yield the full completion once (providers may override).
        yield await self.complete(system, prompt)


class MockLLM(LLMProvider):
    """Deterministic, key-free reasoning for the public demo.

    It does not call any network. It produces clinically-plausible, clearly
    *simulated* text grounded in the structured signal/RAG context it is given,
    so the whole product is testable end-to-end with zero configuration.
    """

    def __init__(self, role: str):
        self.role = role

    async def complete(self, system: str, prompt: str) -> str:
        from .mock_reasoning import mock_response  # local import avoids cycle
        return mock_response(self.role, system, prompt)

    async def stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        text = await self.complete(system, prompt)
        for chunk in _wordwise(text):
            yield chunk


class OpenAICompatibleLLM(LLMProvider):
    def __init__(self, cfg: Resolved):
        self.cfg = cfg

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            h["Authorization"] = f"Bearer {self.cfg.api_key}"
        return h

    def _body(self, system: str, prompt: str, stream: bool) -> dict:
        return {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": stream,
        }

    async def complete(self, system: str, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{self.cfg.base_url}/chat/completions",
                headers=self._headers(),
                json=self._body(system, prompt, stream=False),
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    async def stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.cfg.base_url}/chat/completions",
                headers=self._headers(),
                json=self._body(system, prompt, stream=True),
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        delta = json.loads(payload)["choices"][0]["delta"]
                        if (c := delta.get("content")):
                            yield c
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue


class AnthropicLLM(LLMProvider):
    def __init__(self, cfg: Resolved):
        self.cfg = cfg

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.cfg.api_key or "",
            "anthropic-version": "2023-06-01",
        }

    def _body(self, system: str, prompt: str, stream: bool) -> dict:
        return {
            "model": self.cfg.model,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": stream,
        }

    async def complete(self, system: str, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{self.cfg.base_url}/messages",
                headers=self._headers(),
                json=self._body(system, prompt, stream=False),
            )
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json().get("content", []))

    async def stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.cfg.base_url}/messages",
                headers=self._headers(),
                json=self._body(system, prompt, stream=True),
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        evt = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    if evt.get("type") == "content_block_delta":
                        if (t := evt.get("delta", {}).get("text")):
                            yield t


def get_llm(role: str, override: Optional[dict] = None) -> tuple[LLMProvider, Resolved]:
    cfg = resolve(role, override)
    if not cfg.runs_real:
        return MockLLM(role), cfg
    if cfg.is_anthropic:
        return AnthropicLLM(cfg), cfg
    return OpenAICompatibleLLM(cfg), cfg


def _wordwise(text: str):
    words = text.split(" ")
    for i, w in enumerate(words):
        yield (w if i == 0 else " " + w)


def provider_catalog() -> list[dict]:
    """Expose the provider registry (+ model lists) to the UI for the picker."""
    out = []
    for pid, (base, model, is_anthropic) in PROVIDERS.items():
        out.append(
            {
                "id": pid,
                "default_model": model,
                "models": MODELS.get(pid, [model]),
                "default_base_url": base,
                "local": pid in LOCAL_PROVIDERS,
                "needs_key": pid not in LOCAL_PROVIDERS,
                "kind": "anthropic" if is_anthropic else "openai-compatible",
            }
        )
    return out


async def list_local_models(base_url: str | None = None) -> dict:
    """Probe a local server for its installed models ("local GPU line-up").

    Tries Ollama's GET /api/tags first, then the OpenAI-compatible GET /v1/models
    (vLLM / LM Studio). Returns quickly with whatever is reachable.
    """
    base = (base_url or "http://localhost:11434").rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    found: list[dict] = []
    reachable = False
    # 1) Ollama /api/tags
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{root}/api/tags")
            if r.status_code == 200:
                reachable = True
                for m in r.json().get("models", []):
                    det = m.get("details", {})
                    found.append({
                        "id": m.get("name") or m.get("model"),
                        "size_gb": round(m.get("size", 0) / 1e9, 2),
                        "params": det.get("parameter_size"),
                        "quant": det.get("quantization_level"),
                        "family": det.get("family"),
                        "server": "ollama",
                    })
    except Exception:
        pass
    # 2) OpenAI-compatible /v1/models (vLLM / LM Studio)
    if not found:
        for url in (f"{root}/v1/models", f"{base}/models"):
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    r = await client.get(url)
                    if r.status_code == 200:
                        reachable = True
                        for m in r.json().get("data", []):
                            found.append({"id": m.get("id"), "server": "openai-compatible"})
                        break
            except Exception:
                continue
    return {"reachable": reachable, "base_url": base, "models": found, "count": len(found)}
