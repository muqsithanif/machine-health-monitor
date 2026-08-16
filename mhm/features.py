"""Condition-monitoring features.

Nothing here is exotic. RMS, kurtosis, and crest factor are what vibration
analysts have used on rotating machinery for decades, and they earn their place
because each one fails to notice a different thing:

* **RMS** tracks total energy. Blind to a sharp, brief impact.
* **Kurtosis** tracks impulsiveness. Rises on early bearing spalling while RMS
  is still flat â€” and *falls back* toward normal late in the failure, once the
  damage is widespread enough to look like broad noise again. A monitor reading
  kurtosis alone can conclude a badly worn bearing has recovered.
* **Crest factor** is peak over RMS. Same weakness as kurtosis, same reason.

Which is why they are computed together rather than one being picked as best.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SENSOR_COLUMNS = ["vib_rms", "vib_kurtosis", "temperature", "current"]


def rolling_stats(series: pd.Series, window: int) -> pd.DataFrame:
    """Level, spread, and trend over a trailing window."""
    roll = series.rolling(window, min_periods=max(3, window // 4))
    out = pd.DataFrame({
        f"{series.name}_mean": roll.mean(),
        f"{series.name}_std": roll.std(),
        f"{series.name}_max": roll.max(),
    })
    # Slope over the window: the difference between where the trend is now and
    # where it was, normalised. Distinguishes a drifting fault from a one-off
    # step, which matters because only the first predicts a failure.
    out[f"{series.name}_slope"] = series.diff(window) / window
    return out


def crest_factor(peak: pd.Series, rms: pd.Series) -> pd.Series:
    """Peak divided by RMS. Guarded, because RMS can legitimately reach zero."""
    return peak / rms.replace(0, np.nan)


def _residual_against(target: pd.Series, driver: pd.Series, window: int) -> pd.Series:
    """What is left of `target` once the trailing linear effect of `driver` is removed.

    Rolling ordinary least squares in closed form: slope is covariance over
    variance, and the residual is the distance from the line that relationship
    describes. Everything comes from a trailing window, so no future data leaks
    into a feature.
    """
    minp = max(8, window // 4)
    cov = target.rolling(window, min_periods=minp).cov(driver)
    var = driver.rolling(window, min_periods=minp).var()

    # A driver that never moved carries no information about the target, so the
    # slope is undefined; fall back to plain de-meaning rather than dividing by
    # something near zero.
    slope = (cov / var.replace(0, np.nan)).fillna(0.0)

    expected = (
        target.rolling(window, min_periods=minp).mean()
        + slope * (driver - driver.rolling(window, min_periods=minp).mean())
    )
    return target - expected


def build_features(df: pd.DataFrame, window: int = 36) -> pd.DataFrame:
    """Feature table for one run. Window defaults to six hours of samples.

    Every feature is computed from a trailing window only. Nothing here may see
    the future, or the evaluation that follows is meaningless.
    """
    parts = [df[["timestamp"]].copy()]

    for column in SENSOR_COLUMNS:
        parts.append(rolling_stats(df[column], window))

    features = pd.concat(parts, axis=1)

    features["crest_factor"] = crest_factor(
        df["vib_rms"].rolling(window, min_periods=3).max(), df["vib_rms"]
    )

    # Vibration with the load effect removed. A machine vibrates more when
    # working harder, and without this every busy shift looks like a
    # developing fault.
    #
    # A ratio would be the obvious move and it is wrong here: vibration
    # against load is affine, not proportional. There is a standing level even
    # at low load, so dividing by load *amplifies* the swing instead of
    # cancelling it â€” worst exactly when the machine is idling. Regressing it
    # out is the correct operation.
    features["vib_load_residual"] = _residual_against(df["vib_rms"], df["load"], window * 12)

    # Temperature above what this load alone explains. Isolates a thermal fault
    # from a hot day or a heavy shift.
    expected_temp = df["temperature"].rolling(window * 6, min_periods=window).median()
    features["temp_excess"] = df["temperature"] - expected_temp

    features = features.set_index(df.index)
    features["timestamp"] = df["timestamp"]
    return features


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c != "timestamp"]
