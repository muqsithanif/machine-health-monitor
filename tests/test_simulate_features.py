import numpy as np
import pytest

from mhm.features import SENSOR_COLUMNS, build_features
from mhm.simulate import RunSpec, generate_run


def run(**kw):
    return generate_run(RunSpec("t", hours=300, seed=11, **kw))


def test_healthy_run_has_no_fault_labels():
    df = run(fault_type="none")
    assert not df["faulty"].any()


def test_fault_starts_where_declared_and_never_reverses():
    df = run(fault_type="bearing", fault_onset_fraction=0.5)
    onset = int(np.argmax(df["faulty"].to_numpy()))
    assert onset == pytest.approx(len(df) * 0.5, rel=0.02)
    # Degradation grows; it does not switch on and off.
    progress = df["fault_progress"].to_numpy()
    assert np.all(np.diff(progress) >= -1e-9)


def test_bearing_wear_moves_kurtosis_before_rms():
    # The reason kurtosis is worth computing: early spalling is impulsive long
    # before it raises total energy.
    df = run(fault_type="bearing", fault_onset_fraction=0.5)
    onset = int(np.argmax(df["faulty"].to_numpy()))
    early = slice(onset, onset + (len(df) - onset) // 4)

    base = df.iloc[:onset]
    k_rise = (df["vib_kurtosis"][early].mean() - base["vib_kurtosis"].mean()) / base["vib_kurtosis"].std()
    r_rise = (df["vib_rms"][early].mean() - base["vib_rms"].mean()) / base["vib_rms"].std()
    assert k_rise > r_rise


def test_imbalance_raises_rms_while_leaving_kurtosis_alone():
    df = run(fault_type="imbalance", fault_onset_fraction=0.5)
    base, late = df.iloc[:len(df) // 2], df.iloc[-len(df) // 10:]
    assert late["vib_rms"].mean() > base["vib_rms"].mean() * 1.3
    assert late["vib_kurtosis"].mean() < base["vib_kurtosis"].mean() * 1.3


def test_overheating_moves_temperature_and_current_together():
    df = run(fault_type="overheat", fault_onset_fraction=0.5)
    base, late = df.iloc[:len(df) // 2], df.iloc[-len(df) // 10:]
    assert late["temperature"].mean() > base["temperature"].mean() + 10
    assert late["current"].mean() > base["current"].mean()


def test_transients_exist_and_are_labelled_but_are_not_faults():
    df = run(fault_type="bearing", transient_count=5)
    assert df["is_transient"].sum() == 5
    # A transient inside the healthy period must not be labelled faulty.
    healthy_transients = df["is_transient"] & ~df["faulty"]
    assert healthy_transients.any()


def test_same_seed_reproduces_the_run():
    a, b = run(fault_type="bearing"), run(fault_type="bearing")
    assert np.allclose(a["vib_rms"], b["vib_rms"])


def test_features_never_look_into_the_future():
    # Truncating the run must not change any feature computed before the cut.
    df = run(fault_type="bearing")
    full = build_features(df)
    partial = build_features(df.iloc[:200].copy())

    cols = [c for c in full.columns if c != "timestamp"]
    a = full.iloc[:200][cols].to_numpy()
    b = partial[cols].to_numpy()
    assert np.allclose(a, b, equal_nan=True)


def test_feature_table_covers_every_sensor():
    features = build_features(run(fault_type="bearing"))
    for sensor in SENSOR_COLUMNS:
        assert any(c.startswith(sensor) for c in features.columns)
    assert {"crest_factor", "vib_load_residual", "temp_excess"} <= set(features.columns)


def test_regressing_load_out_leaves_only_sensor_noise():
    # Transients are excluded here on purpose: they are not load-driven, so no
    # amount of regression should remove them and leaving them in would measure
    # the wrong thing. The simulator's noise floor is 0.045.
    df = run(fault_type="none", transient_count=0)
    residual = build_features(df)["vib_load_residual"].dropna()

    assert residual.std() < df["vib_rms"].std() / 1.8
    assert residual.std() == pytest.approx(0.045, abs=0.012)


def test_transients_survive_the_load_regression():
    # And they must. A load-driven swing is not news; an unexplained jump is.
    df = run(fault_type="none", transient_count=6)
    residual = build_features(df)["vib_load_residual"].dropna()
    quiet = build_features(run(fault_type="none", transient_count=0))["vib_load_residual"].dropna()
    assert residual.std() > quiet.std() * 1.5


def test_a_ratio_would_not_have_worked():
    # Guards the reasoning, not just the result. Vibration against load is
    # affine, so dividing by load amplifies the swing instead of cancelling it.
    df = run(fault_type="none", transient_count=0)
    residual = build_features(df)["vib_load_residual"].dropna()
    naive_ratio = (df["vib_rms"] / df["load"].clip(lower=0.1)).loc[residual.index]
    assert residual.std() < naive_ratio.std()


def test_residual_still_rises_when_a_fault_develops():
    # Removing the load effect must not remove the fault signal with it.
    df = run(fault_type="imbalance", fault_onset_fraction=0.5)
    residual = build_features(df)["vib_load_residual"]
    healthy = residual[~df["faulty"]].dropna()
    late = residual.iloc[-len(df) // 10:].dropna()
    assert late.mean() > healthy.mean() + 3 * healthy.std()
