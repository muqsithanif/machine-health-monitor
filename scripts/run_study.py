"""Run every detector over the whole fleet and write the comparison.

    python scripts/run_study.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mhm.detectors import all_detectors  # noqa: E402
from mhm.evaluate import (  # noqa: E402
    evaluate,
    leaderboard,
    summarise,
    threshold_from_healthy,
)
from mhm.features import build_features  # noqa: E402
from mhm.simulate import HOURS_PER_SAMPLE, default_fleet  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
TRAIN_FRACTION = 0.35


def plot_run(spec, raw, features, scored, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    hours = raw.index * HOURS_PER_SAMPLE

    axes[0].plot(hours, raw["vib_rms"], lw=0.8, color="#3d7dd8", label="vibration RMS")
    axes[0].set_ylabel("mm/s")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].plot(hours, raw["vib_kurtosis"], lw=0.8, color="#e67e22", label="kurtosis")
    axes[1].set_ylabel("kurtosis")
    axes[1].legend(loc="upper left", fontsize=8)

    for name, (scores, threshold) in scored.items():
        axes[2].plot(hours, scores, lw=0.9, label=name)
        axes[2].axhline(threshold, ls=":", lw=0.7, alpha=0.5)
    axes[2].set_ylabel("anomaly score")
    axes[2].set_xlabel("hours")
    axes[2].legend(loc="upper left", fontsize=8)
    axes[2].set_yscale("symlog")

    if raw["faulty"].any():
        onset = float(raw.index[raw["faulty"]][0] * HOURS_PER_SAMPLE)
        for ax in axes:
            ax.axvspan(onset, hours.max(), color="#f85149", alpha=0.07)
            ax.axvline(onset, color="#f85149", lw=1, ls="--")
        axes[0].text(onset, axes[0].get_ylim()[1], " fault onset",
                     color="#f85149", fontsize=8, va="top")

    train_end = float(len(raw) * TRAIN_FRACTION * HOURS_PER_SAMPLE)
    for ax in axes:
        ax.axvspan(0, train_end, color="#3fb950", alpha=0.05)
    axes[0].text(0, axes[0].get_ylim()[1], " baseline period",
                 color="#3fb950", fontsize=8, va="top")

    fig.suptitle(spec.name, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    evaluations = []

    for spec, raw in default_fleet():
        features = build_features(raw)
        scored = {}

        for detector in all_detectors():
            scores = detector.fit_score(features, TRAIN_FRACTION)
            threshold = threshold_from_healthy(scores, TRAIN_FRACTION)
            scored[detector.name] = (scores, threshold)
            evaluations.append(
                evaluate(detector.name, spec.name, scores, raw, threshold)
            )

        slug = spec.name.split()[0].replace("-", "_")
        plot_run(spec, raw, features, scored, RESULTS / f"{slug}.png")

    detail = summarise(evaluations)
    board = leaderboard(evaluations)

    detail.to_csv(RESULTS / "per_machine.csv", index=False)
    board.to_csv(RESULTS / "leaderboard.csv", index=False)

    print("\n=== per machine ===")
    print(detail.to_string(index=False))
    print("\n=== leaderboard (mean across fleet) ===")
    print(board.to_string(index=False))
    print(f"\nwritten to {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
