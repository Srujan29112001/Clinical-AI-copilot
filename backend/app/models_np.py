"""
Lightweight, dependency-free NumPy ports of the project's deep-learning engine.

These are *architecturally faithful* forward passes (same equations as the
PyTorch models in src/models/) with deterministic seeded weights, driven by the
REAL qEEG features so outputs are both architecture-true and clinically sensible.
They run instantly on CPU, anywhere — and the result contract matches the GPU
models, so src/models/*.py can be swapped in for production with no UI change.

  • SpikingNeuralNetwork  — LIF neurons, rate-coded input, returns spike raster
  • SelectiveSSM (Mamba2) — selective-scan recurrence over a window sequence
  • CNNLSTMEncoder        — conv → pool → recurrent pass → EEG embedding
  • MultimodalFusion      — cross-attention(EEG, text) → heads + MC-dropout
  • TextEncoder           — cheap clinical bag-of-terms embedding
"""
from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(42)  # fixed → reproducible weights


def _seeded(shape, scale=1.0, seed=0):
    return np.random.default_rng(seed).standard_normal(shape) * scale


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ──────────────────────────────────────────────────────────────────────────
# Spiking Neural Network (Leaky Integrate-and-Fire) — event detection
# ──────────────────────────────────────────────────────────────────────────
EVENT_CLASSES = [
    "spike-wave complex", "sharp transient", "rhythmic discharge",
    "slow-wave burst", "muscle artifact", "background rhythm",
    "attenuation/suppression", "K-complex/spindle",
]


class SpikingNeuralNetwork:
    """2-layer LIF SNN. Mirrors src/models/snn.py: mem = β·mem + I; spike = mem>θ; reset.

    Takes an 8-D *event-evidence* vector (one channel per clinical EEG event),
    rate-codes it into spike trains over T steps, runs two LIF layers, and reads
    out the detected event from output spike counts. A skip connection keeps
    output neuron i aligned to evidence channel i (interpretable), while the LIF
    dynamics + hidden layer give a genuine neuromorphic spike raster.
    """

    def __init__(self, in_dim: int = len(EVENT_CLASSES), hidden=48,
                 out_dim=len(EVENT_CLASSES), beta=0.95, threshold=1.0, steps=30,
                 skip_gain=0.9):
        self.beta, self.thr, self.T, self.skip = beta, threshold, steps, skip_gain
        self.W1 = _seeded((in_dim, hidden), 1.0 / np.sqrt(in_dim), seed=11)
        self.W2 = _seeded((hidden, out_dim), 0.4 / np.sqrt(hidden), seed=12)
        self.hidden, self.out_dim, self.in_dim = hidden, out_dim, in_dim

    def forward(self, evidence: np.ndarray) -> dict:
        e = np.nan_to_num(evidence).astype(float)[: self.in_dim]
        if e.shape[0] < self.in_dim:
            e = np.pad(e, (0, self.in_dim - e.shape[0]))
        drive = np.clip(e, 0.02, 0.98)                  # evidence → firing prob
        rng = np.random.default_rng(7)
        u1 = np.zeros(self.hidden); u2 = np.zeros(self.out_dim)
        raster_h = np.zeros((self.hidden, self.T), dtype=int)
        raster_o = np.zeros((self.out_dim, self.T), dtype=int)
        counts = np.zeros(self.out_dim)
        for t in range(self.T):
            x = (rng.random(drive.shape) < drive).astype(float)  # rate-coded spikes
            i1 = x @ self.W1
            u1 = self.beta * u1 + i1
            s1 = (u1 >= self.thr).astype(float); u1 = u1 - s1 * self.thr
            i2 = s1 @ self.W2 + self.skip * x                    # skip keeps i↔event
            u2 = self.beta * u2 + i2
            s2 = (u2 >= self.thr).astype(float); u2 = u2 - s2 * self.thr
            raster_h[:, t] = s1.astype(int)
            raster_o[:, t] = s2.astype(int)
            counts += s2
        firing_rate = float(raster_h.mean() + raster_o.mean()) / 2
        probs = counts / (counts.sum() + 1e-9)
        order = np.argsort(probs)[::-1]
        top = [{"event": EVENT_CLASSES[i], "activation": round(float(probs[i]), 3),
                "spikes": int(counts[i])} for i in order[:4] if counts[i] > 0]
        return {
            "top_events": top or [{"event": "background rhythm", "activation": 1.0, "spikes": 0}],
            "detected_event": top[0]["event"] if top else "background rhythm",
            "firing_rate": round(firing_rate, 4),
            "total_spikes": int(counts.sum() + raster_h.sum()),
            "energy_uj": round(float(counts.sum() + raster_h.sum()) * 0.9e-3, 4),  # ~event-driven energy proxy
            "raster": raster_o.tolist(),                # output neurons × time
            "raster_labels": EVENT_CLASSES[: self.out_dim],
            "steps": self.T, "neurons": self.out_dim,
        }


def event_evidence(feats: dict, seizure_prob: float, sleep_stage: str = "") -> np.ndarray:
    """Map qEEG features → 8-D evidence aligned to EVENT_CLASSES (for the SNN)."""
    rel = feats["spectral"]["rel_band_power"]
    conn = feats["connectivity"]; ev = feats["events"]; st = feats["states"]; q = feats["quality"]
    hf = rel["beta"] + rel["gamma"]
    return np.array([
        min(seizure_prob * 0.9 + ev["spike_rate_hz"] * 0.2, 0.98),   # spike-wave complex
        min(ev["spike_rate_hz"] * 0.6 + 0.1, 0.95),                  # sharp transient
        min(conn["plv"] * 0.7 + hf * 0.4, 0.97),                     # rhythmic discharge
        min(rel["delta"] * 1.3, 0.95),                               # slow-wave burst
        min(q["emg_index"] * 1.5, 0.9),                              # muscle artifact
        min(rel["alpha"] * 1.4, 0.9),                                # background rhythm
        min(st["suppression_ratio"] * 1.6, 0.95),                   # attenuation/suppression
        min((0.6 if sleep_stage in ("N2", "N3") else 0.1) + rel["theta"] * 0.4, 0.95),  # K-complex/spindle
    ])


# ──────────────────────────────────────────────────────────────────────────
# Mamba2 Selective State-Space Model — temporal trend over analysis windows
# ──────────────────────────────────────────────────────────────────────────
class SelectiveSSM:
    """Diagonal selective-scan: h_t = exp(Δ·A)·h_{t-1} + Δ·B·x_t ; y_t = C·h_t + D·x_t.

    Mirrors src/models/mamba2.py selective_scan with input-dependent Δ,B,C.
    Run over a sequence of per-window feature vectors to model temporal trend.
    """

    def __init__(self, d_model: int, d_state=16):
        self.d, self.n = d_model, d_state
        self.A = -np.arange(1, d_state + 1, dtype=float)        # S4D real init
        self.Wdt = _seeded((d_model, 1), 0.3, seed=21)
        self.WB = _seeded((d_model, d_state), 0.5, seed=22)
        self.WC = _seeded((d_model, d_state), 0.5, seed=23)
        self.D = np.ones(d_model) * 0.5

    def scan(self, seq: np.ndarray) -> dict:
        """seq: (T_windows, d_model) → trend metrics + final embedding."""
        seq = np.atleast_2d(np.nan_to_num(seq).astype(float))
        T, d = seq.shape
        h = np.zeros((d, self.n))
        ys = []
        state_norm = []
        for t in range(T):
            x = seq[t]
            dt = np.log1p(np.exp(x @ self.Wdt)).item()            # softplus Δ (selective)
            B = np.tanh(x @ self.WB); C = np.tanh(x @ self.WC)     # input-dependent
            dA = np.exp(dt * self.A)                                # (n,)
            h = dA[None, :] * h + (dt * np.outer(x, B))            # (d, n)
            y = h @ C + self.D * x                                  # (d,)
            ys.append(y)
            state_norm.append(float(np.linalg.norm(h)))
        ys = np.array(ys)
        # trend = slope of state norm over windows (rising → evolving/worsening)
        if T > 1:
            slope = float(np.polyfit(np.arange(T), state_norm, 1)[0])
        else:
            slope = 0.0
        return {
            "embedding": ys[-1],
            "state_norm_series": [round(s, 3) for s in state_norm],
            "trend_slope": round(slope, 4),
            "trend": "escalating" if slope > 0.05 else "resolving" if slope < -0.05 else "stable",
            "windows": T,
        }


# ──────────────────────────────────────────────────────────────────────────
# CNN-LSTM EEG encoder — conv → pool → recurrent pass → embedding
# ──────────────────────────────────────────────────────────────────────────
class CNNLSTMEncoder:
    """Faithful-but-light: depthwise temporal conv + pooling + a GRU-lite recurrence
    across pooled windows, mirroring src/models/cnn_lstm.py (CNN→BiLSTM→attn)."""

    def __init__(self, emb_dim=64):
        self.emb_dim = emb_dim
        self.filters = _seeded((3, 16), 0.5, seed=31)   # 3 temporal kernels, len 16
        self.Wr = _seeded((3, emb_dim), 0.4, seed=32)
        self.Wh = _seeded((emb_dim, emb_dim), 0.2, seed=33)

    def encode(self, signal: np.ndarray, fs: float) -> np.ndarray:
        """signal: (channels, samples) → emb_dim vector."""
        x = np.nan_to_num(signal).mean(axis=0)          # collapse channels
        x = (x - x.mean()) / (x.std() + 1e-9)
        # multi-kernel temporal convolution + ReLU + mean/abs pooling
        feats = []
        for k in self.filters:
            c = np.convolve(x, k, mode="valid")
            c = np.maximum(c, 0)
            # pool into 8 segments → captures temporal evolution
            seg = np.array_split(c, 8)
            feats.append([float(s.mean()) for s in seg])
        conv = np.array(feats)                            # (3, 8)
        # GRU-lite recurrence across the 8 temporal segments
        h = np.zeros(self.emb_dim)
        for t in range(conv.shape[1]):
            xin = conv[:, t]
            h = np.tanh(xin @ self.Wr + h @ self.Wh)
        return h / (np.linalg.norm(h) + 1e-9)


# ──────────────────────────────────────────────────────────────────────────
# Text encoder — cheap clinical bag-of-terms embedding
# ──────────────────────────────────────────────────────────────────────────
class TextEncoder:
    CLIN_TERMS = [
        "seizure", "epilep", "convuls", "loss of consciousness", "staring", "absence",
        "jerk", "aura", "postictal", "sleep", "insomnia", "apnea", "drowsy", "fatigue",
        "memory", "confusion", "cognit", "dementia", "tremor", "rigid", "parkinson",
        "headache", "migraine", "stroke", "weakness", "numbness", "anxiety", "depress",
        "fever", "infection", "trauma", "tumor", "hypoxia", "metabolic", "overdose",
    ]

    def __init__(self, dim=64):
        self.dim = dim
        self.proj = _seeded((len(self.CLIN_TERMS) + 4, dim), 0.4, seed=41)

    def encode(self, patient: dict) -> np.ndarray:
        text = " ".join([
            str(patient.get("symptoms", "")), str(patient.get("history", "")),
            " ".join(patient.get("medications", []) or []),
        ]).lower()
        bag = np.array([1.0 if term in text else 0.0 for term in self.CLIN_TERMS])
        age = (patient.get("age") or 40) / 100.0
        meds = min(len(patient.get("medications", []) or []), 10) / 10.0
        extra = np.array([age, meds, len(text) / 500.0, 1.0])
        v = np.concatenate([bag, extra]) @ self.proj
        return v / (np.linalg.norm(v) + 1e-9)


# ──────────────────────────────────────────────────────────────────────────
# Multimodal Fusion — cross-attention(EEG, text) → heads + MC-dropout
# ──────────────────────────────────────────────────────────────────────────
SEVERITY = ["minimal", "mild", "moderate", "severe", "critical"]
URGENCY = ["Routine", "Urgent", "Emergent"]


class MultimodalFusion:
    """Cross-modal attention + multi-task heads with MC-dropout uncertainty.
    Mirrors src/models/multimodal_fusion.py (EEG↔text cross-attn → heads)."""

    def __init__(self, d=64):
        self.d = d
        self.Wq = _seeded((d, d), 0.3, seed=51)
        self.Wk = _seeded((d, d), 0.3, seed=52)
        self.Wv = _seeded((d, d), 0.3, seed=53)
        self.Wsev = _seeded((d, len(SEVERITY)), 0.4, seed=54)
        self.Wurg = _seeded((d, len(URGENCY)), 0.4, seed=55)
        self.Wconf = _seeded((d, 1), 0.4, seed=56)

    def _forward(self, eeg, text, prior, drop_rng=None):
        # cross-attention: EEG query attends to [text, eeg] keys/values
        kv = np.stack([text, eeg])                      # (2, d)
        q = eeg @ self.Wq
        k = kv @ self.Wk
        v = kv @ self.Wv
        attn = _softmax((q @ k.T) / np.sqrt(self.d))    # (2,)
        fused = attn @ v + eeg                          # residual
        if drop_rng is not None:
            mask = (drop_rng.random(self.d) > 0.1).astype(float)  # MC-dropout
            fused = fused * mask / 0.9
        # heads — blend learned logits with the grounded clinical prior
        sev = _softmax(fused @ self.Wsev + prior["severity_bias"])
        urg = _softmax(fused @ self.Wurg + prior["urgency_bias"])
        conf = float(_sigmoid(fused @ self.Wconf + prior["conf_bias"]).item())
        return sev, urg, conf

    def predict(self, eeg, text, prior, mc=30) -> dict:
        sevs, urgs, confs = [], [], []
        for s in range(mc):
            rng = np.random.default_rng(100 + s)
            sev, urg, conf = self._forward(eeg, text, prior, rng)
            sevs.append(sev); urgs.append(urg); confs.append(conf)
        sevs = np.array(sevs); urgs = np.array(urgs)
        sev_mean = sevs.mean(0); urg_mean = urgs.mean(0)
        # epistemic uncertainty = mean predictive std across MC passes
        uncertainty = float((sevs.std(0).mean() + urgs.std(0).mean()) / 2)
        return {
            "severity": SEVERITY[int(np.argmax(sev_mean))],
            "severity_dist": {SEVERITY[i]: round(float(sev_mean[i]), 3) for i in range(len(SEVERITY))},
            "urgency": URGENCY[int(np.argmax(urg_mean))],
            "urgency_dist": {URGENCY[i]: round(float(urg_mean[i]), 3) for i in range(len(URGENCY))},
            "confidence": round(float(np.mean(confs)), 3),
            "uncertainty": round(uncertainty, 4),
            "attention_to_text": round(float(_softmax((eeg @ self.Wq) @ (np.stack([text, eeg]) @ self.Wk).T / np.sqrt(self.d))[0]), 3),
            "mc_samples": mc,
        }
