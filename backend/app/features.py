"""
Comprehensive quantitative-EEG (qEEG) feature extraction — pure NumPy/SciPy.

Computes a LARGE, clinically-grounded parameter set per channel and aggregated,
so downstream detectors and the report have rich, explainable inputs:

  Spectral   : abs/rel band power, band ratios (θ/β, DAR α/δ, δ/θ),
               spectral edge frequency (SEF95), median & peak frequency,
               spectral entropy
  Time-domain: RMS, line-length, zero-crossing rate, kurtosis, skewness,
               Hjorth (activity/mobility/complexity)
  Complexity : Shannon, sample, permutation, approximate entropy
  Connectivity: mean Pearson r, alpha-band magnitude-squared coherence,
                phase-locking value (PLV), phase-lag index (PLI)
  Lateralisation: inter-hemispheric asymmetry index (per band)
  Events     : spike / sharp-wave count (line-length z-transients)
  States     : burst-suppression ratio, suppression fraction
  Quality    : flatline %, clipping %, EMG (γ) index, blink (frontal δ) index
  Topography : per-channel band-power map (for the head-map visual)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import signal as sps

BANDS = {
    "delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0), "gamma": (30.0, 50.0),
}
TEN_TWENTY = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3",
              "C3", "Cz", "C4", "T4", "P3", "Pz", "P4", "O1"]
LEFT = {"Fp1", "F7", "F3", "T3", "C3", "P3", "O1", "T5"}
RIGHT = {"Fp2", "F8", "F4", "T4", "C4", "P4", "O2", "T6"}
POSTERIOR = {"O1", "O2", "P3", "P4", "Pz", "P7", "P8", "T5", "T6"}
FRONTAL = {"Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"}


def channel_names(n: int) -> list[str]:
    if n <= len(TEN_TWENTY):
        return TEN_TWENTY[:n]
    return TEN_TWENTY + [f"Ch{i}" for i in range(len(TEN_TWENTY), n)]


# ── spectral ────────────────────────────────────────────────────────────────
def _welch(x: np.ndarray, fs: float):
    nper = int(min(2 * fs, x.shape[-1]))
    f, p = sps.welch(x, fs=fs, nperseg=nper, axis=-1)
    return f, p


def band_power(freqs, psd, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return np.trapz(psd[..., m], freqs[m], axis=-1)


def spectral_metrics(x: np.ndarray, fs: float) -> dict:
    f, p = _welch(x, fs)              # p: (ch, freqs)
    pm = p.mean(axis=0)              # mean across channels
    total = np.trapz(pm, f) + 1e-12
    abs_bp = {b: float(band_power(f, pm, lo, hi)) for b, (lo, hi) in BANDS.items()}
    rel_bp = {b: float(v / total) for b, v in abs_bp.items()}
    # cumulative for SEF95 / median frequency
    csum = np.cumsum(pm) / (np.sum(pm) + 1e-12)
    sef95 = float(f[np.searchsorted(csum, 0.95)]) if len(f) else 0.0
    medf = float(f[np.searchsorted(csum, 0.50)]) if len(f) else 0.0
    peakf = float(f[int(np.argmax(pm))]) if len(f) else 0.0
    pn = pm / (pm.sum() + 1e-12)
    spec_ent = float(-np.sum(pn * np.log2(pn + 1e-12)) / np.log2(len(pn) + 1e-12))
    ratios = {
        "theta_beta": rel_bp["theta"] / (rel_bp["beta"] + 1e-9),
        "delta_alpha": rel_bp["delta"] / (rel_bp["alpha"] + 1e-9),  # DAR
        "delta_theta": rel_bp["delta"] / (rel_bp["theta"] + 1e-9),
        "alpha_delta": rel_bp["alpha"] / (rel_bp["delta"] + 1e-9),
    }
    return {
        "abs_band_power": {k: round(v, 4) for k, v in abs_bp.items()},
        "rel_band_power": {k: round(v, 4) for k, v in rel_bp.items()},
        "band_ratios": {k: round(v, 3) for k, v in ratios.items()},
        "sef95": round(sef95, 2), "median_freq": round(medf, 2),
        "peak_freq": round(peakf, 2), "spectral_entropy": round(spec_ent, 4),
        "_psd_freqs": f, "_psd_perchan": p,
    }


# ── time-domain ───────────────────────────────────────────────────────────────
def hjorth(x: np.ndarray) -> dict:
    d1 = np.diff(x, axis=-1); d2 = np.diff(d1, axis=-1)
    v0 = x.var(axis=-1) + 1e-12; v1 = d1.var(axis=-1) + 1e-12; v2 = d2.var(axis=-1) + 1e-12
    return {
        "activity": float(v0.mean()),
        "mobility": float(np.sqrt(v1 / v0).mean()),
        "complexity": float((np.sqrt(v2 / v1) / np.sqrt(v1 / v0)).mean()),
    }


def time_metrics(x: np.ndarray) -> dict:
    rms = float(np.sqrt((x ** 2).mean()))
    ll = float(np.abs(np.diff(x, axis=-1)).sum(axis=-1).mean())
    zc = float((np.abs(np.diff(np.sign(x), axis=-1)) > 0).sum(axis=-1).mean())
    xf = x.ravel()
    mu, sd = xf.mean(), xf.std() + 1e-12
    z = (xf - mu) / sd
    kurt = float((z ** 4).mean() - 3.0)
    skew = float((z ** 3).mean())
    return {"rms": round(rms, 3), "line_length": round(ll, 2),
            "zero_crossing_rate": round(zc, 2),
            "kurtosis": round(kurt, 3), "skewness": round(skew, 3),
            **{k: round(v, 4) for k, v in hjorth(x).items()}}


# ── complexity / entropy ─────────────────────────────────────────────────────
def shannon_entropy(x: np.ndarray, bins: int = 32) -> float:
    h, _ = np.histogram(x.ravel(), bins=bins, density=True)
    h = h[h > 0]
    p = h / h.sum()
    return float(-np.sum(p * np.log2(p)) / np.log2(bins))


def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    x = x.ravel()
    n = len(x) - (order - 1) * delay
    if n <= 0:
        return 0.0
    patterns = np.empty((n, order), dtype=int)
    for i in range(order):
        patterns[:, i] = x[i * delay: i * delay + n]
    perms = np.argsort(patterns, axis=1)
    _, counts = np.unique(perms, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)) / np.log2(math.factorial(order)))


def sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    x = x.ravel()
    x = (x - x.mean()) / (x.std() + 1e-12)
    if len(x) > 1500:                       # subsample for speed
        x = x[:: len(x) // 1500]
    n = len(x)
    rr = r
    def phi(mm):
        templates = np.array([x[i:i + mm] for i in range(n - mm)])
        if len(templates) < 2:
            return 0.0
        count = 0
        for i in range(len(templates)):
            d = np.max(np.abs(templates - templates[i]), axis=1)
            count += np.sum(d <= rr) - 1
        return count
    a = phi(m + 1); b = phi(m)
    if a == 0 or b == 0:
        return 0.0
    return float(-np.log(a / b))


def approximate_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    x = x.ravel()
    x = (x - x.mean()) / (x.std() + 1e-12)
    if len(x) > 1200:
        x = x[:: len(x) // 1200]
    n = len(x)
    def phi(mm):
        templates = np.array([x[i:i + mm] for i in range(n - mm + 1)])
        c = np.zeros(len(templates))
        for i in range(len(templates)):
            d = np.max(np.abs(templates - templates[i]), axis=1)
            c[i] = np.mean(d <= r)
        return np.mean(np.log(c + 1e-12))
    return float(abs(phi(m) - phi(m + 1)))


def complexity_metrics(x: np.ndarray) -> dict:
    ch0 = x[0]
    return {
        "shannon_entropy": round(shannon_entropy(x), 4),
        "permutation_entropy": round(permutation_entropy(ch0), 4),
        "sample_entropy": round(sample_entropy(ch0), 4),
        "approximate_entropy": round(approximate_entropy(ch0), 4),
    }


# ── connectivity ──────────────────────────────────────────────────────────────
def connectivity(x: np.ndarray, fs: float) -> dict:
    if x.shape[0] < 2:
        return {"mean_correlation": 0.0, "alpha_coherence": 0.0, "plv": 0.0, "pli": 0.0}
    c = np.corrcoef(x)
    iu = np.triu_indices_from(c, k=1)
    mean_corr = float(np.abs(c[iu]).mean())
    # analytic signal for PLV/PLI (band-limited to alpha for stability)
    b, a = sps.butter(4, [8 / (fs / 2), 13 / (fs / 2)], btype="band")
    xa = sps.filtfilt(b, a, x, axis=-1)
    analytic = sps.hilbert(xa, axis=-1)
    phase = np.angle(analytic)
    n = x.shape[0]
    plvs, plis = [], []
    for i in range(n):
        for j in range(i + 1, n):
            dphi = phase[i] - phase[j]
            plvs.append(np.abs(np.mean(np.exp(1j * dphi))))
            plis.append(np.abs(np.mean(np.sign(np.sin(dphi)))))
    # alpha-band coherence via Welch cross-spectra (sampled pairs)
    f, _ = _welch(x[:1], fs)
    coh_vals = []
    pairs = min(n, 6)
    for i in range(pairs):
        for j in range(i + 1, pairs):
            ff, cxy = sps.coherence(x[i], x[j], fs=fs, nperseg=int(min(2 * fs, x.shape[-1])))
            m = (ff >= 8) & (ff < 13)
            coh_vals.append(np.mean(cxy[m]) if m.any() else 0.0)
    return {
        "mean_correlation": round(mean_corr, 4),
        "alpha_coherence": round(float(np.mean(coh_vals)) if coh_vals else 0.0, 4),
        "plv": round(float(np.mean(plvs)) if plvs else 0.0, 4),
        "pli": round(float(np.mean(plis)) if plis else 0.0, 4),
    }


# ── lateralisation / asymmetry ────────────────────────────────────────────────
def asymmetry(x: np.ndarray, fs: float, chans: list[str]) -> dict:
    f, p = _welch(x, fs)
    out = {}
    li = [k for k, c in enumerate(chans) if c in LEFT]
    ri = [k for k, c in enumerate(chans) if c in RIGHT]
    if not li or not ri:
        return {"index": 0.0, "lateralization": "symmetric", "per_band": {}}
    per_band = {}
    overall = []
    for b, (lo, hi) in BANDS.items():
        lp = float(band_power(f, p[li], lo, hi).mean())
        rp = float(band_power(f, p[ri], lo, hi).mean())
        ai = (lp - rp) / (lp + rp + 1e-12)
        per_band[b] = round(ai, 4)
        overall.append(ai)
    idx = float(np.mean(overall))
    lat = "left" if idx > 0.15 else "right" if idx < -0.15 else "symmetric"
    return {"index": round(idx, 4), "lateralization": lat, "per_band": per_band}


# ── events / states / quality ─────────────────────────────────────────────────
def spike_events(x: np.ndarray, fs: float) -> dict:
    win = max(int(0.05 * fs), 4)
    ll = np.abs(np.diff(x, axis=-1))
    # sliding line-length per channel
    kernel = np.ones(win) / win
    counts = 0
    for ch in ll:
        env = np.convolve(ch, kernel, mode="same")
        z = (env - env.mean()) / (env.std() + 1e-12)
        # count threshold crossings with refractory spacing
        peaks, _ = sps.find_peaks(z, height=4.0, distance=int(0.1 * fs))
        counts += len(peaks)
    rate = counts / (x.shape[-1] / fs) / max(x.shape[0], 1)
    return {"spike_count": int(counts), "spike_rate_hz": round(float(rate), 3)}


def burst_suppression(x: np.ndarray, fs: float) -> dict:
    seg = max(int(0.5 * fs), 16)
    amp = np.abs(x).mean(axis=0)
    n = len(amp) // seg
    if n == 0:
        return {"suppression_ratio": 0.0, "burst_suppression": False}
    env = amp[: n * seg].reshape(n, seg).mean(axis=1)
    thr = 0.10 * np.median(env[env > 0]) if np.any(env > 0) else 0
    supp = float(np.mean(env < max(thr, 1e-9)))
    # alternation between high and low → burst-suppression
    high = env > 3 * (thr + 1e-9)
    transitions = int(np.sum(np.abs(np.diff(high.astype(int)))))
    bs = supp > 0.2 and transitions >= 2
    return {"suppression_ratio": round(supp, 3), "burst_suppression": bool(bs)}


def quality(x: np.ndarray, fs: float, spec: dict) -> dict:
    flat = float(np.mean(x.std(axis=-1) < 1e-3))
    mx = np.max(np.abs(x)) + 1e-12
    clip = float(np.mean(np.abs(x) > 0.98 * mx))
    emg = spec["rel_band_power"]["gamma"]          # high-freq → muscle
    f, p = spec["_psd_freqs"], spec["_psd_perchan"]
    # frontal delta as blink proxy
    blink = spec["rel_band_power"]["delta"]
    score = max(0.0, 1.0 - (flat + clip + max(0, emg - 0.25)))
    return {"flatline_frac": round(flat, 3), "clipping_frac": round(clip, 4),
            "emg_index": round(emg, 3), "blink_index": round(blink, 3),
            "quality_score": round(min(score, 1.0), 3)}


def topography(x: np.ndarray, fs: float, chans: list[str]) -> list[dict]:
    f, p = _welch(x, fs)
    out = []
    for k, ch in enumerate(chans):
        tot = np.trapz(p[k], f) + 1e-12
        bp = {b: round(float(band_power(f, p[k], lo, hi) / tot), 4) for b, (lo, hi) in BANDS.items()}
        out.append({"channel": ch, "rel_band_power": bp,
                    "dominant": max(bp, key=bp.get)})
    return out


# ── public API ───────────────────────────────────────────────────────────────
def extract_all(x: np.ndarray, fs: float, chans: list[str] | None = None) -> dict[str, Any]:
    """Full qEEG feature set. x: (channels, samples)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    chans = chans or channel_names(x.shape[0])
    spec = spectral_metrics(x, fs)
    feats = {
        "spectral": {k: v for k, v in spec.items() if not k.startswith("_")},
        "time_domain": time_metrics(x),
        "complexity": complexity_metrics(x),
        "connectivity": connectivity(x, fs),
        "asymmetry": asymmetry(x, fs, chans),
        "events": spike_events(x, fs),
        "states": burst_suppression(x, fs),
        "quality": quality(x, fs, spec),
        "topography": topography(x, fs, chans),
        "channels": chans,
        "n_features": 0,
    }
    # count scalar features for the "parameters" badge
    feats["n_features"] = _count_scalars(feats)
    return feats


def _count_scalars(d: Any) -> int:
    n = 0
    if isinstance(d, dict):
        for k, v in d.items():
            if k in ("channels", "topography"):
                continue
            n += _count_scalars(v)
    elif isinstance(d, (int, float)):
        n += 1
    return n


def feature_vector(feats: dict) -> np.ndarray:
    """Flatten the key scalar features into a fixed vector for the models."""
    s, t, c, cx = feats["spectral"], feats["time_domain"], feats["complexity"], feats["connectivity"]
    vec = [
        *s["rel_band_power"].values(),
        *s["band_ratios"].values(),
        s["sef95"] / 50, s["median_freq"] / 50, s["peak_freq"] / 50, s["spectral_entropy"],
        t["mobility"], t["complexity"], min(t["line_length"] / 5000, 2), t["zero_crossing_rate"] / 500,
        np.tanh(t["kurtosis"] / 5), np.tanh(t["skewness"]),
        c["shannon_entropy"], c["permutation_entropy"], c["sample_entropy"], min(c["approximate_entropy"], 2),
        cx["mean_correlation"], cx["alpha_coherence"], cx["plv"], cx["pli"],
        abs(feats["asymmetry"]["index"]), feats["states"]["suppression_ratio"],
        min(feats["events"]["spike_rate_hz"], 5) / 5, feats["quality"]["quality_score"],
    ]
    return np.asarray(vec, dtype=float)
