"""Anomaly detectors behind one interface, so they can be compared fairly.

Every detector here is fitted on an early slice of the run that is assumed
healthy, then scores the rest. That mirrors commissioning: you have a machine
you believe is fine, you learn what normal looks like, and you watch for
departures.

The comparison exists because "use Isolation Forest" is the reflex answer and
frequently the wrong one. A rolling z-score on the right feature is simpler,
explains itself to a maintenance planner, and on drifting mechanical faults is
often the better warning.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def valid_mask(features: pd.DataFrame) -> np.ndarray:
    """Rows where every feature is available.

    Rolling features are undefined until their window fills, so the first
    samples of a run are NaN. Filling them in — forward, backward, with zero —
    invents data, and the invented values sit at whatever the nearest real
    sample happened to be. That is enough to trip an alarm on sample one and
    make a detector look like it caught a fault weeks early when it caught
    nothing at all.

    So the warm-up is scored NaN and excluded from evaluation instead.
    """
    cols = [c for c in features.columns if c != "timestamp"]
    return ~features[cols].isna().any(axis=1).to_numpy()


class Detector(ABC):
    name: str

    @abstractmethod
    def fit(self, healthy: pd.DataFrame) -> "Detector": ...

    @abstractmethod
    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Higher means more anomalous. NaN where the row cannot be scored.

        Comparable within a detector, not across them.
        """

    def fit_score(self, features: pd.DataFrame, train_fraction: float = 0.35) -> np.ndarray:
        # Fit on the valid part of the baseline window only.
        cut = int(len(features) * train_fraction)
        window = features.iloc[:cut]
        healthy = window[valid_mask(window)]
        if healthy.empty:
            raise ValueError("baseline window contains no fully-formed samples")
        return self.fit(healthy).score(features)


class RollingZScore(Detector):
    """Distance from the healthy baseline, in standard deviations.

    The unglamorous baseline, and the one to beat. Its output is a number a
    planner can act on without trusting a model: 'four sigma above the
    commissioning average' means something on a work order.
    """

    name = "rolling z-score"

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None

    def _cols(self, features: pd.DataFrame) -> list[str]:
        return self.columns or [c for c in features.columns if c != "timestamp"]

    def fit(self, healthy: pd.DataFrame) -> "RollingZScore":
        cols = self._cols(healthy)
        self.mean_ = healthy[cols].mean()
        # Guard against a feature that never moved during commissioning; it
        # would otherwise divide by zero and dominate every score.
        self.std_ = healthy[cols].std().replace(0, np.nan).fillna(1e-6)
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        cols = self._cols(features)
        z = (features[cols] - self.mean_) / self.std_
        # Worst single feature, not the average. Averaging lets three calm
        # sensors hide the one that is screaming.
        scores = z.abs().max(axis=1).to_numpy()
        return np.where(valid_mask(features), scores, np.nan)


class EwmaControlChart(Detector):
    """Exponentially weighted moving average with control limits.

    Built for exactly this problem: catching a small persistent shift that a
    single-sample threshold misses, while ignoring one-off spikes. Standard on
    a process line long before machine learning was involved.
    """

    name = "EWMA control chart"

    def __init__(self, column: str = "vib_rms_mean", alpha: float = 0.12):
        self.column = column
        self.alpha = alpha
        self.mean_ = 0.0
        self.sigma_ = 1.0

    def fit(self, healthy: pd.DataFrame) -> "EwmaControlChart":
        series = healthy[self.column].dropna()
        self.mean_ = float(series.mean())
        self.sigma_ = float(series.std()) or 1e-6
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        mask = valid_mask(features)
        scores = np.full(len(features), np.nan)
        if not mask.any():
            return scores

        # Seed the chart at the fitted mean rather than at the first sample, so
        # it starts in control instead of having to converge into it.
        series = features[self.column].to_numpy()[mask]
        ewma = np.empty(len(series))
        state = self.mean_
        for i, value in enumerate(series):
            state = self.alpha * value + (1 - self.alpha) * state
            ewma[i] = state

        # Steady-state EWMA variance is smaller than the raw signal's by this
        # factor; using the raw sigma would make the chart far too quiet.
        limit_sigma = self.sigma_ * np.sqrt(self.alpha / (2 - self.alpha))
        scores[mask] = np.abs(ewma - self.mean_) / limit_sigma
        return scores


class IsolationForestDetector(Detector):
    """Multivariate outlier score over all features."""

    name = "Isolation Forest"

    def __init__(self, contamination: float = 0.02, seed: int = 42):
        self.contamination = contamination
        self.seed = seed
        self.scaler = StandardScaler()
        self.model: IsolationForest | None = None
        self.columns: list[str] = []

    def fit(self, healthy: pd.DataFrame) -> "IsolationForestDetector":
        self.columns = [c for c in healthy.columns if c != "timestamp"]
        X = healthy[self.columns].to_numpy()
        self.model = IsolationForest(
            n_estimators=200, contamination=self.contamination,
            random_state=self.seed, n_jobs=-1,
        ).fit(self.scaler.fit_transform(X))
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        mask = valid_mask(features)
        scores = np.full(len(features), np.nan)
        if not mask.any():
            return scores
        X = features.loc[mask, self.columns].to_numpy()
        # score_samples is higher for normal points; negate so that, like every
        # other detector here, larger means more anomalous.
        scores[mask] = -self.model.score_samples(self.scaler.transform(X))
        return scores


def all_detectors() -> list[Detector]:
    return [RollingZScore(), EwmaControlChart(), IsolationForestDetector()]
