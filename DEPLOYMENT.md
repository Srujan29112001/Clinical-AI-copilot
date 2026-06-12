# 🚀 Deployment Guide — get a live link

This app has two parts. The **frontend works standalone** (offline demo mode), so
the fastest path to a live URL is just deploying the frontend. Add the backend
when you want the *real* multi-agent pipeline + live local-GPU detection.

> **Why you have to do these steps:** deploying creates resources on **your**
> Vercel / Hugging Face / Render accounts and needs **your** login. I can't log in
> as you, so I've made everything one-click/turnkey — the steps below take ~5–10 min.

---

## ✅ Option A — Frontend only (fastest, ~3 min) → instant live link

The whole app is usable in **offline demo mode** (deterministic engine in the browser:
all 11 agents, 7 detectors, SNN raster, Mamba2, fusion, 3D graph, dataset downloads).

1. Push this repo to **your** GitHub (already done if you forked it).
2. Go to **https://vercel.com/new** → **Import** your `Clinical-AI-copilot` repo.
3. **Root Directory** → click *Edit* → select **`frontend`**. (Framework auto-detects Next.js.)
4. Leave env vars empty (demo mode). Click **Deploy**.
5. ~2 min later you get a live URL like `https://clinical-ai-copilot.vercel.app`. **Done.**

One-click button (replace the repo URL if you forked under a different name):

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FSrujan29112001%2FClinical-AI-copilot&project-name=clinical-ai-copilot&root-directory=frontend)

---

## ✅ Option B — Full stack (real pipeline + local-GPU detection)

Deploy the **backend** first, then point the frontend at it.

### B1. Backend → Hugging Face Spaces (FREE, no cold-starts, best for SSE)

1. Create a free account at **https://huggingface.co**.
2. **New → Space**. Name it e.g. `clinical-ai-backend`. **SDK = Docker** → *Blank*. Create.
3. In the Space, **Files → Add file → Upload files**, and upload the contents of this repo's
   **`backend/`** folder **plus** the repo's **`data/`** folder, keeping this layout:
   ```
   Dockerfile            ← copy from backend/Dockerfile, but change the COPY paths:
   requirements.txt      ← from backend/requirements.txt
   app/                  ← from backend/app/
   data/medical_ontology/   ← from data/medical_ontology/
   data/samples/            ← from data/samples/
   ```
   > Simplest alternative: in the Space settings, set it to **build from your GitHub repo**
   > and use **`backend/Dockerfile`** with build context = repo root (it already copies
   > `backend/app` and `data/medical_ontology`). HF reads `$PORT` (7860) automatically.
4. (Optional) **Settings → Variables and secrets** to enable a real model, e.g.
   `CLINICAL_LLM_PROVIDER=groq` and secret `CLINICAL_LLM_API_KEY=…`. Leave unset for demo reasoning.
5. Also set **`CLINICAL_CORS_ORIGINS`** = your Vercel URL once you have it (step B2).
6. The Space builds and exposes a URL like `https://<you>-clinical-ai-backend.hf.space`.
   Verify: open `…hf.space/health` → `{"status":"ok"}`.

**Alternative backend host — Render** (also free; cold-starts after 15 min idle):
the repo includes [`render.yaml`](render.yaml). On https://render.com → **New → Blueprint** →
connect the repo → it builds `backend/Dockerfile`. Set `CLINICAL_CORS_ORIGINS` to your Vercel URL.

### B2. Frontend → Vercel, pointed at the backend

1. Deploy the frontend as in **Option A** steps 1–3.
2. Before clicking Deploy, add an **Environment Variable**:
   `NEXT_PUBLIC_API_URL = https://<your-backend-url>` (the HF Space or Render URL, **no trailing slash**).
3. Deploy. The navbar will show **“Backend live”** and analyses now run the real FastAPI pipeline,
   with working **local-GPU detection** (if your backend can reach an Ollama/vLLM server).

### B3. Local GPU from the live app

Local-GPU detection probes a server **reachable from the backend**. To use *your* GPU:
run the backend on your own machine (`uvicorn app.main:app --port 8000`) with Ollama running
(`ollama serve`), then set the Vercel `NEXT_PUBLIC_API_URL` to your machine's public URL
(e.g. via a tunnel like `cloudflared` / `ngrok`). In the Studio → Inference settings →
**Ollama (local GPU)** → **Detect models** lists your installed models.

---

## 🔧 Run locally (no deployment)

```bash
# backend
cd backend && python -m venv venv && venv\Scripts\activate   # (mac/linux: source venv/bin/activate)
pip install -r requirements.txt && uvicorn app.main:app --reload --port 8000

# frontend (new terminal)
cd frontend && npm install
echo NEXT_PUBLIC_API_URL=http://localhost:8000 > .env.local
npm run dev      # → http://localhost:3000
```

Or everything in Docker: `docker compose -f docker-compose.app.yml up --build`.

---

## ⚠️ Notes

- **Vercel cannot host the SSE backend** (serverless functions cap at ~300 s and buffer
  `text/event-stream`). Keep the FastAPI backend on HF Spaces / Render / Railway / Fly.
- **CORS:** set `CLINICAL_CORS_ORIGINS` to your exact frontend origin in production
  (the wildcard default disables credentialed CORS, which is fine for this app).
- Decision-support only — **not for clinical use**.
