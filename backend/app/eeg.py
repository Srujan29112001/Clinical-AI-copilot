"""
Dependency-light EEG / dataset processing.

A trimmed, NumPy/SciPy-only reimplementation of the heavy MNE pipeline in
`src/signal_processing/` so the API runs anywhere (incl. serverless) without a
GPU or clinical toolchain. It accepts CSV / NPY / JSON / EDF-ish uploads, runs
bandpass + notch filtering, extracts spectral / connectivity / entropy features,
and produces a transparent, rule-based seizure-probability estimate.

When the full research stack (src/models/cnn_lstm.py etc.) is installed and a
GPU is available, `analyze_array` can be swapped for the deep model — the
feature contract (the returned dict) is identical.
"""
from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
from scipy import signal as sps

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}
DEFAULT_FS = 256.0


# ── loaders ────────────────────────────────────────────────────────────────
def load_signal(filename: str, raw: bytes) -> tuple[np.ndarray, float, dict]:
    """Return (channels x samples float array, fs, meta)."""
    name = (filename or "").lower()
    meta: dict[str, Any] = {"source": filename}
    try:
        if name.endswith(".npy"):
            arr = np.load(io.BytesIO(raw))
        elif name.endswith(".json"):
            obj = json.loads(raw.decode("utf-8", "ignore"))
            arr = np.asarray(obj["data"] if isinstance(obj, dict) else obj, dtype=float)
            if isinstance(obj, dict):
                meta["fs"] = obj.get("fs") or obj.get("sampling_rate")
        elif name.endswith((".csv", ".tsv", ".txt")):
            delim = "\t" if name.endswith(".tsv") else ","
            arr = _read_csv(raw, delim)
        elif name.endswith((".edf", ".bdf")):
            arr, fs = _read_edf(raw)
            meta["fs"] = fs
        else:  # best-effort CSV
            arr = _read_csv(raw, ",")
    except Exception as exc:  # noqa: BLE001 — surface as synthetic fallback
        meta["load_error"] = str(exc)
        arr = _synthetic()

    arr = np.atleast_2d(np.asarray(arr, dtype=float))
    # orient as channels x samples (assume more samples than channels)
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    arr = np.nan_to_num(arr)
    fs = float(meta.get("fs") or DEFAULT_FS)
    meta["channels"] = int(arr.shape[0])
    meta["samples"] = int(arr.shape[1])
    meta["duration_s"] = round(arr.shape[1] / fs, 2)
    return arr, fs, meta


def _read_csv(raw: bytes, delim: str) -> np.ndarray:
    text = raw.decode("utf-8", "ignore").strip().splitlines()
    rows, started = [], False
    for line in text:
        parts = [p.strip() for p in line.split(delim) if p.strip() != ""]
        try:
            rows.append([float(p) for p in parts])
            started = True
        except ValueError:
            if started:  # header in the middle = stop
                break
            continue  # skip leading header
    if not rows:
        raise ValueError("no numeric rows found")
    width = max(len(r) for r in rows)
    rows = [r for r in rows if len(r) == width]
    return np.asarray(rows, dtype=float)


def _read_edf(raw: bytes) -> tuple[np.ndarray, float]:
    try:
        import pyedflib  # type: ignore
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as f:
            f.write(raw)
            path = f.name
        try:
            r = pyedflib.EdfReader(path)
            n = r.signals_in_file
            sigs = np.array([r.readSignal(i) for i in range(n)])
            fs = float(r.getSampleFrequency(0))
            r.close()
            return sigs, fs
        finally:
            os.unlink(path)
    except Exception as exc:  # pyedflib not installed / parse error
        raise ValueError(f"EDF parsing requires pyedflib ({exc})") from exc


def _synthetic(channels: int = 16, seconds: int = 8, fs: float = DEFAULT_FS) -> np.ndarray:
    t = np.arange(int(seconds * fs)) / fs
    rng = np.random.default_rng(42)
    out = []
    for c in range(channels):
        base = (
            0.6 * np.sin(2 * np.pi * 10 * t)   # alpha
            + 0.3 * np.sin(2 * np.pi * 6 * t)  # theta
            + 0.2 * np.sin(2 * np.pi * 22 * t)  # beta
        )
        out.append(base + 0.15 * rng.standard_normal(t.size))
    return np.asarray(out)


# ── preprocessing ──────────────────────────────────────────────────────────
def preprocess(x: np.ndarray, fs: float) -> np.ndarray:
    nyq = fs / 2.0
    hi = min(50.0, nyq * 0.98)
    b, a = sps.butter(4, [0.5 / nyq, hi / nyq], btype="band")
    x = sps.filtfilt(b, a, x, axis=1)
    for f0 in (50.0, 60.0):
        if f0 < nyq:
            bn, an = sps.iirnotch(f0 / nyq, 30.0)
            x = sps.filtfilt(bn, an, x, axis=1)
    return x


# ── features ───────────────────────────────────────────────────────────────
def band_powers(x: np.ndarray, fs: float) -> dict[str, float]:
    freqs, psd = sps.welch(x, fs=fs, nperseg=min(256, x.shape[1]), axis=1)
    psd = psd.mean(axis=0)  # mean across channels
    total = np.trapz(psd, freqs) + 1e-12
    out = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        out[name] = float(np.trapz(psd[mask], freqs[mask]) / total)
    return out


def hjorth(x: np.ndarray) -> dict[str, float]:
    d1 = np.diff(x, axis=1)
    d2 = np.diff(d1, axis=1)
    var0 = x.var(axis=1) + 1e-12
    var1 = d1.var(axis=1) + 1e-12
    var2 = d2.var(axis=1) + 1e-12
    activity = float(var0.mean())
    mobility = float(np.sqrt(var1 / var0).mean())
    complexity = float((np.sqrt(var2 / var1) / np.sqrt(var1 / var0)).mean())
    return {"activity": activity, "mobility": mobility, "complexity": complexity}


def spectral_entropy(x: np.ndarray, fs: float) -> float:
    _, psd = sps.welch(x, fs=fs, nperseg=min(256, x.shape[1]), axis=1)
    psd = psd.mean(axis=0)
    p = psd / (psd.sum() + 1e-12)
    ent = -np.sum(p * np.log2(p + 1e-12))
    return float(ent / np.log2(len(p) + 1e-12))  # normalised 0..1


def line_length(x: np.ndarray) -> float:
    return float(np.abs(np.diff(x, axis=1)).sum(axis=1).mean())


def connectivity(x: np.ndarray) -> float:
    """Mean absolute inter-channel correlation (synchrony proxy)."""
    if x.shape[0] < 2:
        return 0.0
    c = np.corrcoef(x)
    iu = np.triu_indices_from(c, k=1)
    return float(np.abs(c[iu]).mean())


# ── seizure heuristic (transparent, explainable) ───────────────────────────
def seizure_probability(bp: dict[str, float], ent: float, sync: float, ll: float) -> tuple[float, list[str]]:
    """Rule-based, fully explainable estimate in [0,1].

    Ictal EEG is marked by recruiting high-frequency power, hyper-synchrony
    across channels, and rhythmic (low-entropy) high-frequency discharge.
    Slow-wave excess is reported but contributes only mildly (sleep/encephalopathy).
    """
    reasons = []
    score = 0.0
    hf = bp["beta"] + bp["gamma"]
    if hf > 0.5:
        score += 0.45; reasons.append(f"recruiting high-frequency power ({hf:.0%})")
    elif hf > 0.3:
        score += 0.22; reasons.append(f"elevated high-frequency power ({hf:.0%})")
    if sync > 0.8:
        score += 0.35; reasons.append(f"hyper-synchrony across channels ({sync:.2f})")
    elif sync > 0.6:
        score += 0.12; reasons.append(f"raised inter-channel synchrony ({sync:.2f})")
    if hf > 0.4 and ent < 0.5:
        score += 0.10; reasons.append(f"rhythmic high-frequency discharge (entropy {ent:.2f})")
    if bp["delta"] > 0.5:
        score += 0.08; reasons.append(f"excess slow-wave/delta power ({bp['delta']:.0%})")
    if not reasons:
        reasons.append("spectral profile within normal limits")
    return float(min(score, 0.99)), reasons


def analyze_array(x: np.ndarray, fs: float) -> dict[str, Any]:
    xf = preprocess(x, fs)
    bp = band_powers(xf, fs)
    hj = hjorth(xf)
    ent = spectral_entropy(xf, fs)
    sync = connectivity(xf)
    ll = line_length(xf)
    prob, reasons = seizure_probability(bp, ent, sync, ll)
    dominant = max(bp, key=bp.get)
    return {
        "band_powers": {k: round(v, 4) for k, v in bp.items()},
        "dominant_band": dominant,
        "theta_beta_ratio": round(bp["theta"] / (bp["beta"] + 1e-9), 3),
        "hjorth": {k: round(v, 4) for k, v in hj.items()},
        "spectral_entropy": round(ent, 4),
        "synchrony": round(sync, 4),
        "line_length": round(ll, 2),
        "seizure_probability": round(prob, 4),
        "seizure_reasons": reasons,
        "psd_preview": _psd_preview(xf, fs),
        "waveform_preview": _waveform_preview(xf, fs),
    }


def _psd_preview(x: np.ndarray, fs: float, n: int = 64) -> list[dict]:
    freqs, psd = sps.welch(x, fs=fs, nperseg=min(256, x.shape[1]), axis=1)
    psd = psd.mean(axis=0)
    mask = freqs <= 50
    freqs, psd = freqs[mask], psd[mask]
    if len(freqs) > n:
        idx = np.linspace(0, len(freqs) - 1, n).astype(int)
        freqs, psd = freqs[idx], psd[idx]
    psd = psd / (psd.max() + 1e-12)
    return [{"freq": round(float(f), 2), "power": round(float(p), 4)} for f, p in zip(freqs, psd)]


def _waveform_preview(x: np.ndarray, fs: float, seconds: float = 4.0, n: int = 240) -> list[dict]:
    ch = x[0]
    take = min(int(seconds * fs), ch.size)
    seg = ch[:take]
    if seg.size > n:
        idx = np.linspace(0, seg.size - 1, n).astype(int)
        seg = seg[idx]
    seg = seg / (np.abs(seg).max() + 1e-12)
    return [{"t": round(i / len(seg) * seconds, 3), "v": round(float(v), 4)} for i, v in enumerate(seg)]
