# 🧠 Clinical AI Copilot — Multi-Agent EEG Analysis

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Next.js](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-19-149eca.svg)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![Hybrid LLM](https://img.shields.io/badge/inference-local%20GPU%20%2B%20API-22d3ee.svg)](#-hybrid-inference-local-gpu--api)

A **deployable, multi-agent clinical decision-support web app**. Upload an EEG /
clinical dataset and watch **seven specialized AI agents** triage, analyze the
signal, retrieve evidence, diagnose, check drug safety and write a report — in
real time — running on **your local GPU** *or* **any API provider**.

> **v2.0 — full rebuild.** The original repo was a backend-only Python research
> stack with no UI and no way to test it without a 12 GB GPU and a Docker fleet.
> v2 turns it into a modern full-stack product (Next.js + FastAPI) you can run in
> one command, deploy to the web, and use with a hybrid local/cloud inference
> layer — while keeping the deep-learning research engine intact under `src/`.

---

## ✨ Highlights

- **Seven-agent pipeline** — Triage → Signal Analyst → Knowledge Retriever → Diagnostician → Pharmacologist → Safety Critic → Reporter, streamed live over Server-Sent Events.
- **Hybrid inference** — run privately on **Ollama / vLLM / LM Studio** (local GPU, no key) or wire any agent to **Anthropic, OpenAI, Groq, DeepSeek, Mistral, Gemini, OpenRouter**. Mix per agent.
- **Upload your own data** — `EDF · CSV · TSV · NPY · JSON` (channels × samples), or use the built-in ictal / normal / sleep samples.
- **Interactive knowledge graph** — force-directed visualisation of the ICD-10 / SNOMED-CT / RxNorm GraphRAG that grounds every diagnosis.
- **AI chat copilot** — ask questions about the latest analysis (EEG findings, diagnosis, drug safety), grounded in the result.
- **Zero-key live demo** — with no backend configured, the entire app runs a deterministic simulation in the browser, so the public Vercel link is always testable.
- **Explainable seizure detection** — transparent, rule-based estimate from band power, synchrony and entropy (swap in the CNN-LSTM research model for production).

---

## 🏗️ Architecture

```
┌──────────────────────────────┐         SSE          ┌───────────────────────────────┐
│   Frontend — Next.js 15       │  ───────────────────▶│   Backend — FastAPI            │
│   React 19 · Tailwind v4      │  POST /api/analyze   │                                │
│   Framer Motion · Canvas      │  (multipart upload)  │   ┌────────────────────────┐   │
│                               │                      │   │  Multi-agent pipeline   │   │
│   • Landing                   │◀──── data: {stage}───│   │  Triage · Signal ·      │   │
│   • Studio (upload + run)     │◀──── data: {stage}───│   │  Retriever · Dx ·       │   │
│   • AI Chat                   │◀──── data: {result}──│   │  Pharmacologist ·       │   │
│   • Knowledge Graph           │                      │   │  Critic · Reporter      │   │
│   • Architecture              │                      │   └───────────┬────────────┘   │
└──────────────────────────────┘                      │               │                │
        │ no backend?                                  │   ┌───────────▼────────────┐   │
        ▼                                              │   │  Hybrid LLM layer       │   │
  offline simulation                                   │   │  local GPU  OR  API     │   │
  (same event shape)                                   │   └────────────────────────┘   │
                                                       │   EEG DSP (NumPy/SciPy) ·      │
                                                       │   GraphRAG (ICD/SNOMED/Rx) ·   │
                                                       │   drug-interaction engine      │
                                                       └───────────────────────────────┘
```

| Layer | Stack |
|---|---|
| **Frontend** | Next.js 15 · React 19 · TypeScript · Tailwind CSS v4 · Framer Motion · Canvas |
| **Backend** | FastAPI · SSE streaming · async multi-agent orchestrator · httpx |
| **Inference** | Hybrid provider layer — OpenAI-compatible (local + cloud) + native Anthropic |
| **Clinical AI** | In-memory GraphRAG · explainable seizure heuristics · drug-interaction engine |
| **DSP** | NumPy / SciPy (bandpass, notch, Welch PSD, Hjorth, entropy, synchrony) |
| **Research engine** (`src/`) | CNN-LSTM · Transformer · Mamba2 · Llama 3.1 QLoRA · Spiking NN (PyTorch) |

Full design rationale and the original architecture-parameter research are in
[`docs/RESEARCH_DESIGN.md`](docs/RESEARCH_DESIGN.md).

---

## 🚀 Quick start

You need **Node 18+** and **Python 3.10+**. No GPU required.

### 1 · Backend (FastAPI)

```bash
cd backend
python -m venv venv
#  Windows:  venv\Scripts\activate    |   macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 2 · Frontend (Next.js)

```bash
cd frontend
npm install
cp .env.example .env.local        # set NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

Open **http://localhost:3000** → click **Launch Studio** → pick a sample or
upload an EEG → **Run multi-agent analysis**.

> 💡 **Want a zero-setup demo?** Skip the backend entirely. Leave
> `NEXT_PUBLIC_API_URL` empty and `npm run dev` — the app runs the full pipeline
> as an in-browser simulation.

### Or run everything with Docker

```bash
docker compose -f docker-compose.app.yml up --build
# → http://localhost:3000
```

---

## 🔌 Hybrid inference (local GPU + API)

Configure inference per request in the Studio's **Inference settings**, or
server-side via env vars (`backend/.env.example`). The same agent roster can mix
providers.

**Local GPU (private, no key):**
```bash
ollama serve && ollama pull llama3.1:8b
export CLINICAL_LLM_PROVIDER=ollama
export CLINICAL_LLM_BASE_URL=http://localhost:11434/v1
export CLINICAL_LLM_MODEL=llama3.1:8b
```

**Hosted API:**
```bash
export CLINICAL_LLM_PROVIDER=anthropic
export CLINICAL_LLM_API_KEY=sk-ant-...
```

**Hybrid — fast triage local, heavy reasoning in the cloud:**
```bash
export CLINICAL_TRIAGE_PROVIDER=ollama
export CLINICAL_TRIAGE_BASE_URL=http://localhost:11434/v1
export CLINICAL_DIAGNOSTICIAN_PROVIDER=anthropic
export CLINICAL_DIAGNOSTICIAN_API_KEY=sk-ant-...
```

Supported provider ids: `ollama` · `vllm` · `lmstudio` · `local` · `anthropic` ·
`openai` · `groq` · `deepseek` · `mistral` · `gemini` · `openrouter`.

---

## 🌐 Deploy (make it live)

**Frontend → Vercel**
1. Import the repo on [vercel.com](https://vercel.com), set **Root Directory = `frontend`**.
2. (Optional) set `NEXT_PUBLIC_API_URL` to your backend URL. Leave empty for the standalone demo.
3. Deploy. (`frontend/vercel.json` is already configured.)

**Backend → Render / Fly / Railway**
- Render: the repo includes [`render.yaml`](render.yaml) — "New → Blueprint".
- Any Docker host: `docker build -f backend/Dockerfile -t clinical-ai-backend .`
- Set `CLINICAL_CORS_ORIGINS` to your frontend origin in production.

---

## 📁 Project structure

```
Clinical-AI-copilot/
├── frontend/                 # Next.js 15 web app  ← deploy to Vercel
│   ├── app/                  #   landing · studio · chat · graph · architecture
│   ├── components/           #   site · backgrounds · landing · studio · graph · ui
│   └── lib/                  #   api (SSE + upload) · simulate (offline) · agents · store
├── backend/                  # FastAPI multi-agent service  ← deploy to Render/Fly
│   └── app/
│       ├── main.py           #   API routes + SSE
│       ├── agents.py         #   7-agent orchestrator
│       ├── llm.py            #   hybrid local/API provider layer
│       ├── eeg.py            #   NumPy/SciPy signal pipeline
│       ├── knowledge.py      #   in-memory GraphRAG + drug interactions
│       └── samples.py        #   synthetic ictal/normal/sleep datasets
├── src/                      # Deep-learning research engine (PyTorch) — unchanged
├── data/medical_ontology/    # ICD-10 · SNOMED-CT · RxNorm
├── docs/RESEARCH_DESIGN.md   # original architecture-parameter research
├── docker-compose.app.yml    # v2 app stack (frontend + backend)
├── docker-compose.yml        # original full research infra (Neo4j/Qdrant/Kafka/…)
└── render.yaml               # one-click backend deploy
```

---

## 🧬 The research engine (`src/`)

The original deep-learning models are preserved and remain the "advanced engine":
CNN-LSTM seizure detector, Transformer sleep stager, Mamba2 long-context SSM,
Llama 3.1 8B with QLoRA, and a Spiking NN — plus the full HIPAA/streaming/infra
stack. The portable backend exposes the **same result contract**, so you can
swap `backend/app/eeg.py`'s heuristic for the trained CNN-LSTM (and the
in-memory GraphRAG for Neo4j + Qdrant) without touching the UI. See
[`docs/RESEARCH_DESIGN.md`](docs/RESEARCH_DESIGN.md) for every parameter choice.

---

## ⚠️ Disclaimer

**Research and educational use only.** This software is **not FDA-approved** and
must **not** be used for clinical diagnosis or treatment without oversight by
qualified healthcare professionals. The demo / heuristic outputs are clearly
simulated and intended to demonstrate the system architecture.

## 📄 License

MIT — see [LICENSE](LICENSE).

---

**Repository:** [github.com/Srujan29112001/Clinical-AI-copilot](https://github.com/Srujan29112001/Clinical-AI-copilot)
