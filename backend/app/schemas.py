"""Pydantic request/response schemas shared across the API."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────
# LLM configuration (per-request override of server-side env config)
# ──────────────────────────────────────────────────────────────────────────
class LLMConfig(BaseModel):
    """Per-request LLM override sent from the UI.

    `provider` selects the backend. For a *local GPU* model, use
    provider="ollama" (or "vllm"/"local") and point base_url at your local
    server, e.g. http://localhost:11434/v1 — no api_key needed.
    For a hosted provider, set provider + api_key.
    """
    provider: Optional[str] = Field(
        None,
        description="anthropic | openai | groq | deepseek | mistral | "
        "openrouter | gemini | ollama | vllm | local | mock",
    )
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1024


class PatientContext(BaseModel):
    patient_id: str = "DEMO-PATIENT"
    age: Optional[int] = None
    sex: Optional[str] = None
    symptoms: str = ""
    history: Optional[str] = None
    medications: list[str] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    """Used when JSON is posted without a file (sample-dataset runs)."""
    patient: PatientContext = Field(default_factory=PatientContext)
    dataset_id: Optional[str] = None
    llm: LLMConfig = Field(default_factory=LLMConfig)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    patient: Optional[PatientContext] = None
    analysis: Optional[dict[str, Any]] = None
    llm: LLMConfig = Field(default_factory=LLMConfig)


# ──────────────────────────────────────────────────────────────────────────
# Streaming event envelope (Server-Sent Events payloads)
# ──────────────────────────────────────────────────────────────────────────
class AgentEvent(BaseModel):
    type: Literal["stage", "log", "token", "result", "error", "done"]
    agent: Optional[str] = None
    status: Optional[Literal["pending", "running", "done", "error"]] = None
    message: Optional[str] = None
    data: Optional[dict[str, Any]] = None
