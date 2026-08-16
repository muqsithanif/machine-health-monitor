"""Scoring detectors the way a maintenance department would.

Accuracy is the wrong measure here and it is worth being explicit about why. In
a run that is healthy for two thirds of its length, a detector that never fires
scores about 65% accuracy and is completely useless. Precision and recall are
better but still miss the point.

**Lead time is the point.** A detector that flags a bearing three hours before
seizure is technically correct and operationally worthless — nobody schedules a
crew, orders a part, or plans downtime in three hours. One that warns four days
early with a few more false alarms is the better tool, and no accuracy-style
metric will ever say so.

So every detector is reported with lead time first, and false alarms per week
alongside it, because that trade is the actual decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mhm.simulate import HOURS_PER_SAMPLE


@dataclass(frozen=True)
class Evaluation:
    detector: str
    machine: str
    threshold: float
    detected: bool
    lead_time_hours: float | None      # before fault onset ... None if missed
    precision: float
    recall: float
    false_alarms_before_onset: int
    false_alarms_per_week: float
    flagged_transients: int

    def as_row(self) -> dict:
        return {
            "detector": self.detector,
            "machine": self.machine,
            "detected": self.detected,
            "lead_time_h": round(self.lead_time_hours, 1) if self.lead_time_hours is not None else None,
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "false_alarms/week": round(self.false_alarms_per_week, 2),
            "flagged_transients": self.flagged_transients,
        }


def threshold_from_healthy(scores: np.ndarray, healthy_fraction: float, quantile: float = 0.995) -> float:
    """Set the alarm level from the commissioning period alone.

    Choosing it from the whole run — including the failure — would be tuning on
    the answer, and every detector would look better than it is.
    """
    cut = max(10, int(len(scores) * healthy_fraction))
    window = scores[:cut]
    window = window[~np.isnan(window)]
    if window.size == 0:
        raise ValueError("no scored samples in the baseline window")
    return float(np.quantile(window, quantile))


def _first_sustained(flags: np.ndarray, consecutive: int) -> int | None:
    """Index where `consecutive` flags first occur in a row.

    Requiring persistence is what separates a fault from a transient. A single
    sample over the line is noise; six in a row is a machine changing.
    """
    if consecutive <= 1:
        hits = np.flatnonzero(flags)
        return int(hits[0]) if hits.size else None

    run = 0
    for i, flag in enumerate(flags):
        run = run + 1 if flag else 0
        if run >= consecutive:
            return i - consecutive + 1
    return None


def evaluate(
    detector_name: str,
    machine: str,
    scores: np.ndarray,
    truth: pd.DataFrame,
    threshold: float,
    consecutive: int = 6,
) -> Evaluation:
    faulty = truth["faulty"].to_numpy()
    transient = truth["is_transient"].to_numpy()

    # NaN means the row could not be scored — the rolling windows had not
    # filled yet. Treating it as zero would silently count the warm-up as
    # "confirmed healthy"; treating it as a flag would invent an early alarm.
    scored = ~np.isnan(scores)
    flags = np.zeros(len(scores), dtype=bool)
    flags[scored] = scores[scored] >= threshold

    onset = int(np.argmax(faulty)) if faulty.any() else len(faulty)
    alarm_at = _first_sustained(flags, consecutive)

    detected = alarm_at is not None and (not faulty.any() or alarm_at >= 0)
    lead_time = None
    if faulty.any() and alarm_at is not None:
        # Positive means the alarm came before onset; the usual case is a
        # negative value, meaning it fired that many hours into the fault.
        lead_time = (onset - alarm_at) * HOURS_PER_SAMPLE

    # Confusion counts only over rows that were actually scored.
    tp = int((flags & faulty & scored).sum())
    fp = int((flags & ~faulty & scored).sum())
    fn = int((~flags & faulty & scored).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    before_onset = int((flags[:onset]).sum())
    scored_before = int(scored[:onset].sum())
    weeks = max(scored_before * HOURS_PER_SAMPLE / (24 * 7), 1e-6)

    return Evaluation(
        detector=detector_name,
        machine=machine,
        threshold=threshold,
        detected=alarm_at is not None,
        lead_time_hours=lead_time,
        precision=precision,
        recall=recall,
        false_alarms_before_onset=before_onset,
        false_alarms_per_week=before_onset / weeks,
        flagged_transients=int((flags & transient & ~faulty).sum()),
    )


def summarise(evaluations: list[Evaluation]) -> pd.DataFrame:
    df = pd.DataFrame([e.as_row() for e in evaluations])
    return df.sort_values(["machine", "detector"]).reset_index(drop=True)


def leaderboard(evaluations: list[Evaluation]) -> pd.DataFrame:
    """Per-detector averages across the fleet, ordered by what matters."""
    rows = []
    for name in dict.fromkeys(e.detector for e in evaluations):
        subset = [e for e in evaluations if e.detector == name]
        with_fault = [e for e in subset if e.lead_time_hours is not None]
        leads = [e.lead_time_hours for e in with_fault if e.lead_time_hours is not None]
        rows.append({
            "detector": name,
            "faults_detected": f"{sum(1 for e in with_fault if e.detected)}/{len(with_fault)}",
            "mean_lead_time_h": round(float(np.mean(leads)), 1) if leads else None,
            "mean_precision": round(float(np.mean([e.precision for e in subset])), 3),
            "mean_recall": round(float(np.mean([e.recall for e in subset])), 3),
            "false_alarms/week": round(float(np.mean([e.false_alarms_per_week for e in subset])), 2),
            "transients_flagged": sum(e.flagged_transients for e in subset),
        })
    return pd.DataFrame(rows).sort_values("mean_lead_time_h", ascending=False).reset_index(drop=True)
