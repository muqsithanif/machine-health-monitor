"""Generate machine sensor data with known, injected degradation.

Real run-to-failure datasets are scarce and their failure moments are often
disputed. Simulating instead means the ground truth is exact — which is the
only way to measure lead time honestly, because you have to know when the fault
actually started.

The degradation patterns follow how rotating machinery genuinely fails:

* **Bearing wear** — a slow rise in high-frequency vibration energy, then a
  sharper knee near the end. Kurtosis moves before RMS does, because early
  spalling produces impulsive spikes long before it raises overall energy.
* **Imbalance** — vibration grows at the running frequency with the square of
  the fault severity, and stays sinusoidal rather than impulsive.
* **Progressive overheating** — temperature drifts up while current rises with
  it, since a tightening bearing draws more torque.

Each run also gets a handful of isolated spikes that are *not* faults. A
detector that flags those is producing false alarms, and a study that leaves
them out flatters every method equally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SAMPLE_INTERVAL_MINUTES = 10
HOURS_PER_SAMPLE = SAMPLE_INTERVAL_MINUTES / 60


@dataclass(frozen=True)
class RunSpec:
    name: str
    hours: int = 720                 # 30 days at 10-minute sampling
    fault_type: str = "bearing"      # bearing | imbalance | overheat | none
    fault_onset_fraction: float = 0.6
    severity: float = 1.0
    transient_count: int = 6         # non-fault spikes
    seed: int = 42


def _fault_progress(n: int, onset_idx: int) -> np.ndarray:
    """0 before onset, rising 0->1 after it. Faults grow, they do not switch on."""
    progress = np.zeros(n)
    if onset_idx >= n:
        return progress
    remaining = n - onset_idx
    progress[onset_idx:] = np.linspace(0.0, 1.0, remaining)
    return progress


def generate_run(spec: RunSpec) -> pd.DataFrame:
    rng = np.random.default_rng(spec.seed)
    n = int(spec.hours / HOURS_PER_SAMPLE)
    t = np.arange(n)
    onset_idx = int(n * spec.fault_onset_fraction) if spec.fault_type != "none" else n
    progress = _fault_progress(n, onset_idx)

    # --- healthy baseline -------------------------------------------------
    # Load cycles daily; everything else rides on that.
    daily = np.sin(2 * np.pi * t * HOURS_PER_SAMPLE / 24)
    load = 0.65 + 0.20 * daily + rng.normal(0, 0.02, n)

    vib_rms = 1.8 + 0.55 * load + rng.normal(0, 0.045, n)
    vib_kurtosis = 3.0 + rng.normal(0, 0.12, n)   # 3.0 is Gaussian
    temperature = 42.0 + 18.0 * load + rng.normal(0, 0.35, n)
    current = 21.0 + 14.0 * load + rng.normal(0, 0.18, n)

    # --- degradation ------------------------------------------------------
    if spec.fault_type == "bearing":
        # Impulsiveness leads energy: kurtosis climbs early, RMS follows with a
        # knee. This ordering is the whole reason kurtosis is worth computing.
        vib_kurtosis += spec.severity * 6.5 * progress**0.55
        vib_rms += spec.severity * 2.4 * progress**2.2
        temperature += spec.severity * 5.0 * progress**2.0
        current += spec.severity * 1.6 * progress**2.0

    elif spec.fault_type == "imbalance":
        # Grows with the square of severity, and stays sinusoidal — so RMS
        # rises while kurtosis barely moves.
        vib_rms += spec.severity * 3.1 * progress**2
        vib_kurtosis += spec.severity * 0.25 * progress
        current += spec.severity * 2.2 * progress**2

    elif spec.fault_type == "overheat":
        temperature += spec.severity * 22.0 * progress**1.5
        current += spec.severity * 4.5 * progress**1.5
        vib_rms += spec.severity * 0.5 * progress

    # --- transients that are not faults -----------------------------------
    # Load steps, a passing forklift, a sensor glitch. Short and self-clearing.
    transient_idx = rng.choice(n, size=spec.transient_count, replace=False)
    for idx in transient_idx:
        width = rng.integers(1, 4)
        end = min(idx + width, n)
        vib_rms[idx:end] += rng.uniform(0.8, 1.5)
        vib_kurtosis[idx:end] += rng.uniform(1.5, 3.0)

    timestamps = pd.date_range("2026-01-01", periods=n, freq=f"{SAMPLE_INTERVAL_MINUTES}min")

    return pd.DataFrame({
        "timestamp": timestamps,
        "vib_rms": vib_rms,
        "vib_kurtosis": vib_kurtosis,
        "temperature": temperature,
        "current": current,
        "load": load,
        # Ground truth, for evaluation only. Never fed to a detector.
        "faulty": progress > 0,
        "fault_progress": progress,
        "is_transient": np.isin(t, transient_idx),
    })


def default_fleet() -> list[tuple[RunSpec, pd.DataFrame]]:
    """A small fleet covering each failure mode plus a healthy control."""
    specs = [
        RunSpec("pump-01 bearing wear", fault_type="bearing", severity=1.0, seed=1),
        RunSpec("pump-02 bearing wear (mild)", fault_type="bearing", severity=0.55, seed=2),
        RunSpec("fan-01 imbalance", fault_type="imbalance", severity=1.0, seed=3),
        RunSpec("compressor-01 overheating", fault_type="overheat", severity=1.0, seed=4),
        RunSpec("pump-03 healthy", fault_type="none", seed=5),
    ]
    return [(s, generate_run(s)) for s in specs]
