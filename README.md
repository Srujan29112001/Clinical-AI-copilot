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

## ✨ Highlights (v3)

- **Eleven-agent pipeline** — Signal Analyst → Neuromorphic Detector → Multi-Detector → Temporal Modeler → Triage → Knowledge Retriever → Multimodal Fusion → Diagnostician → Pharmacologist → Safety Critic → Reporter, streamed live over SSE.
- **Real model engine (NumPy ports of the PyTorch models)** — a **CNN-LSTM** EEG encoder, a **Spiking Neural Network** (LIF) that emits a live **spike raster**, a **Mamba2** selective-state-space temporal model, and a **multimodal cross-attention fusion** head with **Monte-Carlo-dropout uncertainty**. Faithful forward passes driven by the *real* features — same result contract as `src/models/`.
- **50+ qEEG parameters** — band powers & ratios, spectral edge frequency, 4 entropies (Shannon/sample/permutation/approximate), Hjorth, connectivity (Pearson/coherence/PLV/PLI), inter-hemispheric asymmetry, spike rate, burst-suppression ratio, signal-quality metrics, per-electrode topography.
- **7 detection use-cases** — seizure (generalized vs focal, lateralised), sleep stage (W/N1/N2/N3/REM), encephalopathy (DAR), burst-suppression, focal abnormality, posterior dominant rhythm, recording quality.
- **Hybrid inference, per agent** — local GPU (**Ollama / vLLM / LM Studio**, no key) *or* **Anthropic · OpenAI · Groq · DeepSeek · Mistral · Gemini · OpenRouter**, with a **per-provider model picker** and **live local-GPU model detection** built into the app.
- **3D interactive knowledge graph** — three.js force-directed GraphRAG (ICD-10 / SNOMED-CT / RxNorm) with click-for-description, plus a 2D mode.
- **Upload your own data** — `EDF · CSV · TSV · NPY · JSON`, or download the **6 built-in test datasets** (see below).
- **AI chat copilot** grounded in the latest analysis · **zero-key offline demo** so the public Vercel link always works.

## 🧪 Test datasets

Six realistic 16-channel (10-20 montage) EEG recordings ship in [`data/samples/`](data/samples/) and are downloadable from the Studio:

| File | Condition | Expected result |
|---|---|---|
| `eeg_normal_awake.csv` | Healthy, eyes-closed alpha | Routine · PDR present |
| `eeg_seizure_generalized.csv` | 3 Hz spike-wave + HF recruitment | **Emergent** · generalized seizure |
| `eeg_seizure_focal_left_temporal.csv` | Left temporal discharge | Urgent · focal seizure, **left** lateralization |
| `eeg_sleep_n2.csv` | N2 sleep with spindles | Routine · sleep staging |
| `eeg_encephalopathy_diffuse_slowing.csv` | Diffuse delta (high DAR) | encephalopathy |
| `eeg_burst_suppression.csv` | Burst-suppression | **Emergent** · high suppression ratio |

Regenerate / extend them with `python scripts/generate_test_datasets.py`.

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

**Full step-by-step (with exactly what you do manually): [DEPLOYMENT.md](DEPLOYMENT.md).**

Fastest path — the frontend works standalone (offline demo), so one click gives a live link:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FSrujan29112001%2FClinical-AI-copilot&project-name=clinical-ai-copilot&root-directory=frontend)

- **Frontend → Vercel** — import repo, **Root Directory = `frontend`**, Deploy. (Optional `NEXT_PUBLIC_API_URL` → backend.)
- **Backend → Hugging Face Spaces** (free, no cold-starts, best for SSE) or **Render** (`render.yaml`). Set `CLINICAL_CORS_ORIGINS` to your frontend origin.

> A live URL is created on **your** Vercel/HF account and needs your login — I can't authenticate as you, so deployment is the one manual step. It takes ~5 minutes; [DEPLOYMENT.md](DEPLOYMENT.md) has the clicks.

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
