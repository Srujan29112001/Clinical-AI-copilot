"""
Multi-use-case clinical detectors.

Beyond a single seizure score, this runs SEVEN parallel detectors over the rich
qEEG feature set + per-channel signal, each returning a calibrated score, a
boolean flag, and an explanation grounded in specific parameters:

  1. Seizure (ictal)        — generalized vs focal, lateralised
  2. Sleep stage            — Wake / N1 / N2 / N3 / REM
  3. Encephalopathy         — diffuse slowing (DAR)
  4. Burst-suppression      — anaesthetic/anoxic pattern
  5. Focal abnormality      — lateralised slowing/discharge
  6. Posterior dominant rhythm (PDR) — normal-variant marker
  7. Recording quality      — artifact / usability
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

from .features import POSTERIOR


def _per_channel_focal(signal: np.ndarray, fs: float, chans: list[str]) -> dict:
    """Find the channel with the strongest transient (line-length z-score)."""
    ll = np.abs(np.diff(signal, axis=-1)).sum(axis=-1)
    z = (ll - ll.mean()) / (ll.std() + 1e-9)
    idx = int(np.argmax(z))
    return {"channel": chans[idx] if idx < len(chans) else f"Ch{idx}",
            "z": round(float(z[idx]), 2), "focal": bool(z[idx] > 2.0)}


def detect_all(signal: np.ndarray, fs: float, feats: dict) -> dict:
    chans = feats["channels"]
    rel = feats["spectral"]["rel_band_power"]
    ratios = feats["spectral"]["band_ratios"]
    conn = feats["connectivity"]
    asym = feats["asymmetry"]
    states = feats["states"]
    events = feats["events"]
    quality = feats["quality"]
    medf = feats["spectral"]["median_freq"]
    peakf = feats["spectral"]["peak_freq"]

    hf = rel["beta"] + rel["gamma"]
    focal = _per_channel_focal(signal, fs, chans)
    detections = []

    # 1 ── Seizure -----------------------------------------------------------
    sz = 0.0
    why = []
    if hf > 0.5: sz += 0.4; why.append(f"recruiting high-frequency power ({hf:.0%})")
    elif hf > 0.3: sz += 0.2; why.append(f"elevated high-frequency power ({hf:.0%})")
    if conn["plv"] > 0.7: sz += 0.25; why.append(f"high phase-locking (PLV {conn['plv']:.2f})")
    if conn["mean_correlation"] > 0.8: sz += 0.15; why.append(f"hyper-synchrony ({conn['mean_correlation']:.2f})")
    if events["spike_rate_hz"] > 0.5: sz += 0.2; why.append(f"frequent spikes ({events['spike_rate_hz']:.2f}/s)")
    sz = min(sz, 0.99)
    seizure_type = ("generalized" if conn["mean_correlation"] > 0.75 and sz > 0.5
                    else "focal" if (focal["focal"] and sz > 0.35) else "none")
    detections.append({
        "id": "seizure", "label": "Ictal seizure activity", "score": round(sz, 3),
        "present": sz >= 0.45, "severity_hint": "critical" if sz >= 0.8 else "moderate",
        "detail": {"type": seizure_type, "lateralization": asym["lateralization"],
                   "focal_channel": focal["channel"] if focal["focal"] else None},
        "explanation": "; ".join(why) or "no ictal pattern",
    })

    # 2 ── Sleep stage -------------------------------------------------------
    spindle = _spindle_power(signal, fs)
    stage, sleep_conf, swhy = _sleep_stage(rel, spindle, conn)
    detections.append({
        "id": "sleep", "label": f"Sleep stage: {stage}", "score": round(sleep_conf, 3),
        "present": stage != "Wake", "severity_hint": "minimal",
        "detail": {"stage": stage, "spindle_power": round(spindle, 3)},
        "explanation": swhy,
    })

    # 3 ── Encephalopathy (diffuse slowing) ----------------------------------
    dar = ratios["delta_alpha"]
    enc = min(_ramp(dar, 1.5, 6.0) * 0.7 + _ramp(rel["delta"], 0.35, 0.6) * 0.3, 0.99)
    if spindle > 0.04 or stage in ("N2", "N3"):
        enc *= 0.3  # slowing in sleep is physiological, not encephalopathic
    detections.append({
        "id": "encephalopathy", "label": "Diffuse slowing / encephalopathy",
        "score": round(enc, 3), "present": enc >= 0.5, "severity_hint": "moderate",
        "detail": {"DAR": round(dar, 2), "median_freq": medf},
        "explanation": f"delta/alpha ratio {dar:.1f}, median freq {medf:.1f} Hz"
        if enc >= 0.5 else f"DAR {dar:.1f} within normal range",
    })

    # 4 ── Burst-suppression -------------------------------------------------
    bs = states["suppression_ratio"]
    detections.append({
        "id": "burst_suppression", "label": "Burst-suppression",
        "score": round(min(bs * 1.5, 0.99), 3),
        "present": bool(states["burst_suppression"]), "severity_hint": "critical",
        "detail": {"suppression_ratio": bs},
        "explanation": f"suppression ratio {bs:.0%}" if states["burst_suppression"]
        else "continuous background",
    })

    # 5 ── Focal abnormality / lateralization --------------------------------
    foc = min(abs(asym["index"]) * 2 + (0.3 if focal["focal"] else 0), 0.99)
    detections.append({
        "id": "focal", "label": "Focal abnormality / lateralization",
        "score": round(foc, 3), "present": foc >= 0.4, "severity_hint": "moderate",
        "detail": {"lateralization": asym["lateralization"],
                   "asymmetry_index": asym["index"], "channel": focal["channel"]},
        "explanation": f"{asym['lateralization']} lateralization (AI {asym['index']:+.2f})"
        if foc >= 0.4 else "symmetric, no focal feature",
    })

    # 6 ── Posterior dominant rhythm (normal marker) -------------------------
    pdr_freq, pdr_present = _pdr(signal, fs, chans)
    detections.append({
        "id": "pdr", "label": "Posterior dominant rhythm",
        "score": round(0.9 if pdr_present else 0.1, 3), "present": pdr_present,
        "severity_hint": "minimal",
        "detail": {"frequency_hz": pdr_freq},
        "explanation": f"PDR present at {pdr_freq:.1f} Hz (normal variant)"
        if pdr_present else "no organized posterior alpha",
    })

    # 7 ── Recording quality -------------------------------------------------
    q = quality["quality_score"]
    detections.append({
        "id": "quality", "label": "Recording quality",
        "score": round(q, 3), "present": q < 0.6, "severity_hint": "minimal",
        "detail": quality,
        "explanation": (f"reduced quality (EMG {quality['emg_index']:.0%}, "
                        f"flatline {quality['flatline_frac']:.0%})") if q < 0.6
        else "good signal quality",
    })

    # primary finding = highest-scoring *abnormal* detector
    abnormal = [d for d in detections if d["id"] not in ("pdr", "quality", "sleep") and d["present"]]
    if not abnormal:
        sleep_d = next((d for d in detections if d["id"] == "sleep"), None)
        primary = sleep_d if sleep_d and sleep_d["present"] else None
    else:
        primary = max(abnormal, key=lambda d: d["score"])

    return {
        "detections": detections,
        "primary_finding": primary["id"] if primary else "normal",
        "seizure_probability": round(sz, 3),
        "seizure_type": seizure_type,
        "focal": focal,
    }


# ── helpers ──────────────────────────────────────────────────────────────────
def _ramp(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo + 1e-9), 0, 1))


def _spindle_power(signal: np.ndarray, fs: float) -> float:
    """Relative power in the sleep-spindle band (11–16 Hz)."""
    f, p = sps.welch(signal, fs=fs, nperseg=int(min(2 * fs, signal.shape[-1])), axis=-1)
    pm = p.mean(axis=0)
    total = np.trapz(pm, f) + 1e-12
    m = (f >= 11) & (f < 16)
    return float(np.trapz(pm[m], f[m]) / total)


def _sleep_stage(rel, spindle, conn):
    # N3 (deep sleep) has delta dominance WITH residual theta; pure monomorphic
    # delta with negligible theta is pathological slowing (→ encephalopathy), not N3.
    if rel["delta"] > 0.5 and rel["theta"] > 0.12:
        return "N3", 0.8, f"delta-dominant ({rel['delta']:.0%}) with mixed theta → deep slow-wave sleep"
    if rel["theta"] > 0.4 and spindle > 0.04:
        return "N2", 0.75, f"theta background + spindles ({spindle:.0%})"
    if rel["theta"] > 0.45:
        return "N1", 0.6, f"theta-dominant ({rel['theta']:.0%}), drowsy"
    if rel["beta"] > 0.25 and rel["alpha"] < 0.2:
        return "REM", 0.55, "low-voltage mixed-frequency, REM-like"
    if rel["alpha"] > 0.35:
        return "Wake", 0.85, f"alpha-dominant ({rel['alpha']:.0%}), awake"
    return "Wake", 0.5, "mixed frequencies"


def _pdr(signal: np.ndarray, fs: float, chans: list[str]):
    post_idx = [i for i, c in enumerate(chans) if c in POSTERIOR]
    if not post_idx:
        post_idx = list(range(max(0, len(chans) - 3), len(chans)))
    x = signal[post_idx]
    f, p = sps.welch(x, fs=fs, nperseg=int(min(2 * fs, x.shape[-1])), axis=-1)
    pm = p.mean(axis=0)
    am = (f >= 8) & (f < 13)
    if not am.any():
        return 0.0, False
    alpha_peak = f[am][int(np.argmax(pm[am]))]
    alpha_rel = np.trapz(pm[am], f[am]) / (np.trapz(pm, f) + 1e-12)
    return round(float(alpha_peak), 1), bool(alpha_rel > 0.25)
