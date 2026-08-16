import numpy as np
import pandas as pd
import pytest

from mhm.detectors import (
    EwmaControlChart,
    IsolationForestDetector,
    RollingZScore,
    all_detectors,
    valid_mask,
)
from mhm.features import build_features
from mhm.simulate import RunSpec, generate_run


@pytest.fixture(scope="module")
def bearing_run():
    raw = generate_run(RunSpec("t", hours=400, fault_type="bearing", seed=7))
    return raw, build_features(raw)


def test_valid_mask_excludes_the_warmup(bearing_run):
    _, features = bearing_run
    mask = valid_mask(features)
    assert mask[0] == False          # noqa: E712 — rolling windows unfilled
    assert mask[-1] == True          # noqa: E712
    assert mask.sum() > len(mask) * 0.8


@pytest.mark.parametrize("detector", all_detectors(), ids=lambda d: d.name)
def test_scores_are_nan_exactly_where_features_are_missing(detector, bearing_run):
    _, features = bearing_run
    scores = detector.fit_score(features)
    assert np.array_equal(~np.isnan(scores), valid_mask(features))


@pytest.mark.parametrize("detector", all_detectors(), ids=lambda d: d.name)
def test_a_developing_fault_scores_higher_than_the_baseline(detector, bearing_run):
    raw, features = bearing_run
    scores = detector.fit_score(features)
    healthy = np.nanmean(scores[~raw["faulty"].to_numpy()])
    late = np.nanmean(scores[int(len(scores) * 0.9):])
    assert late > healthy


def test_zscore_uses_the_worst_feature_not_the_average():
    # Averaging lets calm sensors hide the one that is screaming.
    healthy = pd.DataFrame({"a": np.zeros(50), "b": np.zeros(50), "c": np.zeros(50)})
    healthy += np.random.default_rng(0).normal(0, 1, (50, 3))

    d = RollingZScore().fit(healthy)
    one_extreme = pd.DataFrame({"a": [20.0], "b": [0.0], "c": [0.0]})
    assert d.score(one_extreme)[0] > 10


def test_zscore_survives_a_feature_that_never_moved():
    # A sensor stuck at one value during commissioning has zero variance and
    # would otherwise divide by zero and dominate every later score.
    healthy = pd.DataFrame({"a": np.random.default_rng(0).normal(0, 1, 40), "flat": np.ones(40)})
    scores = RollingZScore().fit(healthy).score(healthy)
    assert np.isfinite(scores).all()


def test_ewma_starts_in_control_rather_than_converging_into_it():
    # Seeding the chart at the first sample instead of the fitted mean makes it
    # alarm on startup — which is what made the first version of this study
    # report eight days of "lead time" it had not earned.
    healthy = pd.DataFrame({"vib_rms_mean": np.full(60, 2.0) + np.random.default_rng(1).normal(0, 0.05, 60)})
    d = EwmaControlChart(column="vib_rms_mean").fit(healthy)

    steady = pd.DataFrame({"vib_rms_mean": np.full(30, 2.0)})
    scores = d.score(steady)
    assert scores[0] < 1.0


def test_a_spike_decays_while_a_shift_persists():
    # The property that matters is duration, not peak. A large enough spike can
    # out-peak a small shift for a sample or two — but it falls back, and the
    # evaluator requires consecutive exceedances before calling it an alarm.
    baseline = np.full(120, 2.0)
    d = EwmaControlChart(column="v", alpha=0.12).fit(
        pd.DataFrame({"v": baseline + np.random.default_rng(2).normal(0, 0.05, 120)})
    )

    spike = baseline.copy()
    spike[60] = 6.0                 # one violent sample
    shift = baseline.copy()
    shift[60:] += 0.4               # a small permanent step

    limit = 6.0
    over_spike = int((d.score(pd.DataFrame({"v": spike})) >= limit).sum())
    over_shift = int((d.score(pd.DataFrame({"v": shift})) >= limit).sum())

    assert over_shift > over_spike * 3


def test_a_spike_returns_to_normal_within_the_chart_memory():
    baseline = np.full(120, 2.0)
    d = EwmaControlChart(column="v", alpha=0.12).fit(
        pd.DataFrame({"v": baseline + np.random.default_rng(3).normal(0, 0.05, 120)})
    )
    spike = baseline.copy()
    spike[60] = 6.0
    scores = d.score(pd.DataFrame({"v": spike}))

    assert scores[60] > 10          # it does react
    assert scores[110] < 2          # and it lets go


def test_isolation_forest_fits_and_scores_without_imputation(bearing_run):
    _, features = bearing_run
    scores = IsolationForestDetector().fit_score(features)
    assert np.isfinite(scores[valid_mask(features)]).all()


def test_fit_refuses_a_baseline_with_no_complete_samples():
    features = pd.DataFrame({"a": [np.nan] * 20, "b": [np.nan] * 20})
    with pytest.raises(ValueError):
        RollingZScore().fit_score(features, train_fraction=0.5)
