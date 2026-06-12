"""
Generate realistic, downloadable sample EEG datasets for testing the
Clinical AI Copilot. Produces multichannel CSVs (samples x channels, with a
header row of 10-20 montage channel names) for several clinical conditions.

Run:  python scripts/generate_test_datasets.py
Output: data/samples/*.csv  (+ a manifest.json)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

FS = 256
DUR = 12  # seconds
CH = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3",
      "C3", "Cz", "C4", "T4", "P3", "Pz", "P4", "O1"]
LEFT = {"Fp1", "F7", "F3", "T3", "C3", "P3", "O1"}
OUT = Path(__file__).resolve().parents[1] / "data" / "samples"


def pink(n, rng):
    w = rng.standard_normal(n)
    f = np.fft.rfft(w)
    fr = np.fft.rfftfreq(n)
    fr[0] = fr[1] if len(fr) > 1 else 1.0
    f = f / np.sqrt(fr)
    p = np.fft.irfft(f, n=n)
    return p / (np.abs(p).max() + 1e-9)


def spikes(t, freq, width=0.04):
    ph = (t * freq) % 1.0
    return (ph < width).astype(float)


def gen(kind, rng):
    t = np.arange(int(DUR * FS)) / FS
    n = t.size
    out = []
    for i, ch in enumerate(CH):
        ph = rng.uniform(0, 2 * np.pi, 5)
        amp = rng.uniform(0.8, 1.2)
        bg = pink(n, rng)
        posterior = ch in {"O1", "O2", "P3", "P4", "Pz"}
        if kind == "normal":
            # posterior-dominant alpha, attenuates anteriorly
            a = 1.1 if posterior else 0.5
            sig = a * np.sin(2 * np.pi * 10 * t + ph[0]) + 0.25 * np.sin(2 * np.pi * 18 * t + ph[1]) + 0.9 * bg
        elif kind == "ictal_generalized":
            sig = (0.7 * np.sin(2 * np.pi * 3 * t) + 1.4 * spikes(t, 3.0)
                   + 1.2 * np.sin(2 * np.pi * 24 * t) + 0.8 * np.sin(2 * np.pi * 33 * t))
            sig = amp * sig + 0.08 * bg
        elif kind == "ictal_focal":
            # focal temporal-left rhythmic discharge; rest near-normal
            focal = ch in {"T3", "F7", "C3"}
            if focal:
                sig = (1.3 * spikes(t, 4.0) + 1.0 * np.sin(2 * np.pi * 20 * t) + 0.5 * np.sin(2 * np.pi * 5 * t)) + 0.1 * bg
            else:
                sig = 0.6 * np.sin(2 * np.pi * 9 * t + ph[0]) + 0.9 * bg
            sig *= amp
        elif kind == "sleep_n2":
            # theta background + sleep spindles (~13 Hz bursts) + K-complex-ish slow
            spindle = np.sin(2 * np.pi * 13 * t) * (((t % 3) < 0.7).astype(float))
            sig = 0.9 * np.sin(2 * np.pi * 5 * t + ph[0]) + 0.5 * spindle + 0.3 * np.sin(2 * np.pi * 1.5 * t) + 0.9 * bg
            sig *= amp
        elif kind == "encephalopathy":
            # diffuse delta slowing (high DAR), low alpha
            sig = 1.3 * np.sin(2 * np.pi * 2.0 * t + ph[0]) + 0.4 * np.sin(2 * np.pi * 3.5 * t + ph[1]) + 1.0 * bg
            sig *= amp
        elif kind == "burst_suppression":
            # alternating bursts and flat suppression (~ every 2s)
            env = ((t % 2.0) < 0.6).astype(float)
            burst = (1.4 * np.sin(2 * np.pi * 8 * t) + 1.0 * spikes(t, 6.0) + 0.8 * np.sin(2 * np.pi * 20 * t))
            sig = amp * (burst * env) + 0.05 * bg
        else:
            sig = bg
        out.append(sig.astype(np.float32))
    arr = np.array(out).T  # samples x channels
    # realistic scale to microvolts
    arr = arr / (np.abs(arr).max() + 1e-9) * 80.0
    return arr


DATASETS = {
    "eeg_normal_awake": ("normal", "Healthy adult, eyes-closed posterior dominant alpha. Expect: Routine."),
    "eeg_seizure_generalized": ("ictal_generalized", "Generalized 3 Hz spike-wave + HF recruitment. Expect: Emergent seizure."),
    "eeg_seizure_focal_left_temporal": ("ictal_focal", "Left temporal focal rhythmic discharge. Expect: focal seizure, left lateralization."),
    "eeg_sleep_n2": ("sleep_n2", "Stage N2 sleep with spindles. Expect: sleep staging, Routine."),
    "eeg_encephalopathy_diffuse_slowing": ("encephalopathy", "Diffuse delta slowing, high DAR. Expect: encephalopathy."),
    "eeg_burst_suppression": ("burst_suppression", "Burst-suppression pattern. Expect: high suppression ratio, Emergent."),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, (kind, desc) in DATASETS.items():
        rng = np.random.default_rng(abs(hash(name)) % (2**32))
        arr = gen(kind, rng)
        path = OUT / f"{name}.csv"
        header = ",".join(CH)
        np.savetxt(path, arr, delimiter=",", header=header, comments="", fmt="%.3f")
        manifest.append({
            "file": f"{name}.csv", "condition": kind, "description": desc,
            "channels": len(CH), "samples": int(arr.shape[0]),
            "duration_s": DUR, "sampling_rate": FS, "units": "microvolts",
        })
        print(f"  wrote {path.name:48s} {arr.shape[0]}x{arr.shape[1]}  ({kind})")
    (OUT / "manifest.json").write_text(json.dumps({"datasets": manifest, "montage": CH, "fs": FS}, indent=2))
    print(f"\n{len(manifest)} datasets written to {OUT}")


if __name__ == "__main__":
    main()
