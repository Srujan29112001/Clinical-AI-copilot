"""Built-in sample EEG datasets so the Studio is testable without an upload."""
from __future__ import annotations

import numpy as np

from .eeg import DEFAULT_FS

SAMPLES = {
    "ictal": {
        "id": "ictal",
        "name": "Ictal seizure episode",
        "description": "16-channel scalp EEG with rhythmic spike-wave discharge.",
        "patient": {"patient_id": "DEMO-ICTAL", "age": 34, "sex": "female",
                    "symptoms": "recurrent generalized seizures with loss of consciousness",
                    "medications": ["levetiracetam", "lamotrigine"]},
    },
    "normal": {
        "id": "normal",
        "name": "Normal awake EEG",
        "description": "Healthy adult, eyes-closed posterior dominant alpha rhythm.",
        "patient": {"patient_id": "DEMO-NORMAL", "age": 28, "sex": "male",
                    "symptoms": "routine screening, no acute complaints",
                    "medications": []},
    },
    "sleep": {
        "id": "sleep",
        "name": "Sleep / drowsy EEG",
        "description": "Theta-dominant drowsy recording with reduced synchrony.",
        "patient": {"patient_id": "DEMO-SLEEP", "age": 51, "sex": "female",
                    "symptoms": "excessive daytime sleepiness, suspected sleep disorder",
                    "medications": ["melatonin"]},
    },
}


def synth(kind: str, channels: int = 16, seconds: int = 8, fs: float = DEFAULT_FS) -> np.ndarray:
    """Generate clinically-flavoured multichannel EEG with realistic per-channel
    phase/amplitude variation and 1/f background, so benign recordings read benign
    and the ictal case reads emergent."""
    t = np.arange(int(seconds * fs)) / fs
    rng = np.random.default_rng({"ictal": 1, "normal": 2, "sleep": 3}.get(kind, 0))
    out = []
    for c in range(channels):
        ph = rng.uniform(0, 2 * np.pi, 4)          # per-channel phase
        amp = rng.uniform(0.7, 1.3)                # per-channel amplitude
        bg = _pink(t.size, rng)                    # 1/f background per channel
        if kind == "ictal":
            # globally synchronous, rhythmic, high-frequency-rich → seizure-like
            sig = (0.7 * np.sin(2 * np.pi * 3.0 * t)
                   + 1.4 * sps_square(t, 3.0)
                   + 1.3 * np.sin(2 * np.pi * 24 * t)
                   + 0.9 * np.sin(2 * np.pi * 34 * t))
            sig = amp * sig + 0.08 * bg            # tight coupling across channels
        elif kind == "sleep":
            sig = (0.9 * np.sin(2 * np.pi * 6 * t + ph[0])     # theta
                   + 0.3 * np.sin(2 * np.pi * 2 * t + ph[1])   # slow
                   + 0.9 * bg)
            sig *= amp
        else:  # normal awake — posterior alpha + broadband background
            sig = (0.8 * np.sin(2 * np.pi * 10 * t + ph[0])    # alpha
                   + 0.25 * np.sin(2 * np.pi * 18 * t + ph[1])  # low beta
                   + 1.0 * bg)
            sig *= amp
        out.append(sig)
    return np.asarray(out)


def _pink(n: int, rng) -> np.ndarray:
    """Approximate 1/f (pink) noise via spectral shaping of white noise."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    spec = spec / np.sqrt(freqs)
    pink = np.fft.irfft(spec, n=n)
    return pink / (np.abs(pink).max() + 1e-9)


def sps_square(t: np.ndarray, freq: float) -> np.ndarray:
    """Sparse spike train at `freq` Hz (avoids importing scipy.signal.square)."""
    phase = (t * freq) % 1.0
    return (phase < 0.05).astype(float)
