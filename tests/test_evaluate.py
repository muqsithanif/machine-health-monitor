import numpy as np
import pandas as pd
import pytest

from mhm.evaluate import _first_sustained, evaluate, threshold_from_healthy
from mhm.simulate import HOURS_PER_SAMPLE


def truth_frame(n, onset=None, transients=()):
    faulty = np.zeros(n, dtype=bool)
    if onset is not None:
        faulty[onset:] = True
    is_transient = np.zeros(n, dtype=bool)
    for i in transients:
        is_transient[i] = True
    return pd.DataFrame({"faulty": faulty, "is_transient": is_transient})


def test_single_flag_is_not_an_alarm():
    # One sample over the line is noise. Persistence is what separates a fault
    # from a passing forklift.
    flags = np.zeros(50, dtype=bool)
    flags[10] = True
    assert _first_sustained(flags, consecutive=6) is None


def test_sustained_run_reports_its_start():
    flags = np.zeros(50, dtype=bool)
    flags[20:30] = True
    assert _first_sustained(flags, consecutive=6) == 20


def test_broken_run_does_not_count():
    flags = np.zeros(50, dtype=bool)
    flags[[10, 11, 12, 14, 15, 16]] = True   # gap at 13
    assert _first_sustained(flags, consecutive=6) is None


def test_lead_time_is_positive_when_the_alarm_precedes_the_fault():
    n, onset = 200, 100
    scores = np.zeros(n)
    scores[70:] = 10.0                      # alarms 30 samples early
    ev = evaluate("d", "m", scores, truth_frame(n, onset), threshold=5.0)
    assert ev.lead_time_hours == pytest.approx(30 * HOURS_PER_SAMPLE)


def test_lead_time_is_negative_when_the_alarm_lags_the_fault():
    n, onset = 200, 100
    scores = np.zeros(n)
    scores[130:] = 10.0
    ev = evaluate("d", "m", scores, truth_frame(n, onset), threshold=5.0)
    assert ev.lead_time_hours == pytest.approx(-30 * HOURS_PER_SAMPLE)


def test_a_detector_that_never_fires_is_not_credited():
    n, onset = 200, 100
    ev = evaluate("d", "m", np.zeros(n), truth_frame(n, onset), threshold=5.0)
    assert ev.detected is False
    assert ev.lead_time_hours is None
    assert ev.recall == 0.0


def test_nan_warmup_is_neither_a_flag_nor_a_clean_bill():
    # The bug this guards: filling NaN with zero counts warm-up as confirmed
    # healthy; filling it forward can invent an alarm on sample one.
    n, onset = 200, 100
    scores = np.full(n, np.nan)
    scores[50:] = 0.0
    scores[120:] = 10.0

    ev = evaluate("d", "m", scores, truth_frame(n, onset), threshold=5.0)
    assert ev.lead_time_hours == pytest.approx(-20 * HOURS_PER_SAMPLE)
    # The 50 unscored samples must not appear in the confusion counts.
    assert ev.false_alarms_before_onset == 0


def test_false_alarm_rate_uses_only_scored_time():
    n, onset = 200, 100
    scores = np.full(n, np.nan)
    scores[50:] = 0.0
    scores[60:66] = 10.0                     # one burst before onset

    ev = evaluate("d", "m", scores, truth_frame(n, onset), threshold=5.0)
    scored_weeks = 50 * HOURS_PER_SAMPLE / (24 * 7)
    assert ev.false_alarms_per_week == pytest.approx(6 / scored_weeks)


def test_transients_flagged_are_counted_separately():
    n, onset = 200, 150
    scores = np.zeros(n)
    scores[40:50] = 10.0
    ev = evaluate("d", "m", scores, truth_frame(n, onset, transients=[42, 43]), threshold=5.0)
    assert ev.flagged_transients == 2


def test_threshold_comes_from_the_baseline_not_the_whole_run():
    scores = np.concatenate([np.random.default_rng(0).normal(1, 0.1, 300), np.full(700, 99.0)])
    th = threshold_from_healthy(scores, healthy_fraction=0.3)
    # If the failure period leaked into the threshold it would be near 99.
    assert th < 3.0


def test_threshold_ignores_nan_warmup():
    scores = np.concatenate([np.full(20, np.nan), np.full(280, 2.0), np.full(700, 99.0)])
    assert threshold_from_healthy(scores, 0.3) == pytest.approx(2.0)


def test_threshold_needs_something_to_work_with():
    with pytest.raises(ValueError):
        threshold_from_healthy(np.full(100, np.nan), 0.3)
