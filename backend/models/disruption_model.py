"""
disruption_model.py — FairChain Disruption Engine
==================================================
Defines, trains, and persists the two-tier ML stack used to score
supply-chain route segments for disruption risk.

Architecture
------------
Tier 1  —  IsolationForest (sklearn)
    Real-time anomaly scorer on live feature vectors.
    Contamination is derived dynamically from segment historical volatility.
    Output: isolation_forest_raw_score + normalized_risk_probability [0,1].

Tier 2  —  Facebook Prophet (time-series forecasting)
    Fits a per-segment additive model on a rolling delay history window.
    Incorporates:
      • rainfall_mm_6h as an external additive regressor (monsoon accuracy)
      • Three India-specific custom seasonalities:
          - indian_peak_hours  : morning (07-10 h) + evening (17-21 h) rush
          - indian_festival    : Diwali / Dussehra / Pongal traffic clusters
          - indian_weekend     : Fri-Sat elevated freight movement
    Produces per-horizon (6 h / 12 h / 24 h) delay probability forecasts
    that the route endpoint can attach to the /predict response for forward
    planning in the Mapbox timeline layer.

Tier 3  —  MinMaxScaler
    Normalises Tier-1 raw scores for the agreed [0.0, 1.0] probability output.

Owned by: ML Lead (Disruption Engine)
DO NOT MODIFY outside the 4 ML-owned files.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from data.feature_engineering import (
    FEATURE_COLS,
    LiveSignals,
    build_feature_vector,
    generate_chennai_flood_timeline,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
MODEL_DIR = _HERE / "artifacts"
IF_MODEL_PATH = MODEL_DIR / "isolation_forest.joblib"
SCALER_PATH = MODEL_DIR / "if_scaler.joblib"

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------

# Dynamic contamination bounds (fraction of anomalous samples in the corpus).
# Low-volatility segments (stable corridors) sit near CONTAMINATION_MIN.
# High-volatility segments (flood / accident-prone stretches) approach CONTAMINATION_MAX.
CONTAMINATION_MIN: float = 0.05
CONTAMINATION_MAX: float = 0.12

# Reference variance bounds used to interpolate contamination.
# Derived empirically from the synthetic training distribution:
#   _generate_training_data() draws historical_delay_var ~ Uniform(1, 80)
# Values outside this range are clamped before interpolation.
_VAR_LOW: float = 1.0    # variance → CONTAMINATION_MIN (very stable segment)
_VAR_HIGH: float = 80.0  # variance → CONTAMINATION_MAX (highly volatile segment)

# Base IF params — contamination is injected dynamically at train / score time.
IF_PARAMS: dict = {
    "n_estimators": 200,
    "max_samples": "auto",
    # contamination is NOT set here; use derive_contamination() before constructing the model
    "max_features": 1.0,
    "bootstrap": False,
    "random_state": 42,
    "n_jobs": -1,
    "warm_start": False,
}

# Confidence interval half-width (±) as a linear function of anomaly score
# Wider intervals in the "uncertain" mid-range [0.3, 0.7]
_CI_BASE_HALF_WIDTH = 0.05
_CI_MID_ZONE_BOOST = 0.08


# ---------------------------------------------------------------------------
# Dynamic contamination derivation
# ---------------------------------------------------------------------------
def derive_contamination(historical_delay_variance: float) -> float:
    """
    Map a segment's ``historical_delay_variance`` to an IsolationForest
    contamination value in [CONTAMINATION_MIN, CONTAMINATION_MAX].

    Rationale
    ---------
    A high delay variance indicates an inherently volatile corridor (e.g.,
    a coastal highway like NH48 that floods seasonally).  On such segments a
    larger fraction of historical observations are genuinely anomalous, so
    the IsolationForest should be calibrated with a higher contamination to
    avoid under-counting true anomalies.

    Conversely, a low-variance inland highway (NH44 through the Deccan
    plateau) has a stable baseline; a tighter contamination prevents the
    model from over-flagging routine variability.

    Mapping (linear interpolation, clamped):
        variance ≤ _VAR_LOW  → CONTAMINATION_MIN (0.05)
        variance ≥ _VAR_HIGH → CONTAMINATION_MAX (0.12)
        in-between           → linear blend

    Parameters
    ----------
    historical_delay_variance : float
        Pre-computed delay variance for the highway segment (from the
        segment metadata / Supabase ``route_segments`` table).
        Must be ≥ 0; negative values are treated as 0.

    Returns
    -------
    float
        Contamination value clamped to [CONTAMINATION_MIN, CONTAMINATION_MAX].

    Examples
    --------
    >>> derive_contamination(1.0)   # very stable
    0.05
    >>> derive_contamination(80.0)  # highly volatile
    0.12
    >>> derive_contamination(40.5)  # mid-range
    0.085
    """
    var = max(0.0, historical_delay_variance)
    # Normalise to [0, 1] within the reference range, then clamp
    t = (var - _VAR_LOW) / (_VAR_HIGH - _VAR_LOW)
    t = float(np.clip(t, 0.0, 1.0))
    contamination = CONTAMINATION_MIN + t * (CONTAMINATION_MAX - CONTAMINATION_MIN)
    logger.debug(
        "derive_contamination(var=%.2f) → t=%.4f → contamination=%.4f",
        var, t, contamination,
    )
    return round(contamination, 6)


# ---------------------------------------------------------------------------
# Synthetic training data generator
# ---------------------------------------------------------------------------
def _generate_training_data(n_normal: int = 4000, n_anomaly: int = 340) -> pd.DataFrame:
    """
    Generate a synthetic training dataset representative of Indian NH corridor
    traffic/weather patterns.

    Normal distribution: typical monsoon + dry season operational data.
    Anomaly distribution: flood/landslide/accident scenarios.
    """
    rng = np.random.default_rng(42)

    def _normal_row() -> dict:
        return {
            "rainfall_mm_1h":         rng.uniform(0, 15),
            "rainfall_mm_6h":         rng.uniform(0, 50),
            "velocity_kmh":           rng.uniform(35, 80),
            "velocity_deviation_pct": rng.uniform(0, 20),
            "incident_count_2h":      float(rng.integers(0, 3)),
            "visibility_km":          rng.uniform(2, 12),
            "water_level_m":          rng.uniform(0.2, 1.5),
            "segment_load_factor":    rng.uniform(0.2, 1.0),
            "delay_z_score":          rng.normal(0, 1),
            "temp_celsius":           rng.uniform(20, 42),
            "wind_speed_kmh":         rng.uniform(5, 30),
            "is_night":               float(rng.integers(0, 2)),
            "distance_km":            rng.uniform(10, 120),
            "historical_delay_var":   rng.uniform(1, 50),
        }

    def _anomaly_row() -> dict:
        """Simulate flood / severe disruption conditions."""
        return {
            "rainfall_mm_1h":         rng.uniform(35, 120),
            "rainfall_mm_6h":         rng.uniform(100, 350),
            "velocity_kmh":           rng.uniform(0, 18),
            "velocity_deviation_pct": rng.uniform(45, 100),
            "incident_count_2h":      float(rng.integers(5, 15)),
            "visibility_km":          rng.uniform(0.05, 0.8),
            "water_level_m":          rng.uniform(2.0, 5.0),
            "segment_load_factor":    rng.uniform(1.5, 4.0),
            "delay_z_score":          rng.uniform(2.5, 8.0),
            "temp_celsius":           rng.uniform(20, 32),
            "wind_speed_kmh":         rng.uniform(50, 120),
            "is_night":               float(rng.integers(0, 2)),
            "distance_km":            rng.uniform(10, 120),
            "historical_delay_var":   rng.uniform(1, 80),
        }

    rows: list[dict] = (
        [_normal_row() for _ in range(n_normal)]
        + [_anomaly_row() for _ in range(n_anomaly)]
    )
    df = pd.DataFrame(rows, columns=FEATURE_COLS)
    logger.debug("Training data shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def train_and_save_model(
    training_df: Optional[pd.DataFrame] = None,
    model_path: Path = IF_MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    historical_delay_variance: Optional[float] = None,
) -> tuple[IsolationForest, MinMaxScaler]:
    """
    Train an IsolationForest on the supplied (or auto-generated) DataFrame
    and persist both the model and the MinMaxScaler to disk via joblib.

    Parameters
    ----------
    training_df : pd.DataFrame, optional
        If None, synthetic training data is generated automatically.
    model_path : Path
        Destination for the serialised IsolationForest.
    scaler_path : Path
        Destination for the serialised MinMaxScaler.
    historical_delay_variance : float, optional
        Segment-level delay variance used to derive the IsolationForest
        contamination dynamically via ``derive_contamination()``.
        When None, the contamination is computed from the mean
        ``historical_delay_var`` column of ``training_df`` (or the synthetic
        dataset), so the model still adapts to the data it was trained on
        rather than falling back to any fixed value.

    Returns
    -------
    (IsolationForest, MinMaxScaler)
    """
    if training_df is None:
        logger.info("No training data provided — generating synthetic dataset.")
        training_df = _generate_training_data()

    X = training_df[FEATURE_COLS].values

    # --- Derive contamination from historical volatility ---
    if historical_delay_variance is not None:
        contamination = derive_contamination(historical_delay_variance)
    else:
        # Fall back to the mean variance across the training corpus so
        # the model is still segment-aware even without an explicit value.
        mean_var = float(training_df["historical_delay_var"].mean())
        contamination = derive_contamination(mean_var)
        logger.info(
            "historical_delay_variance not supplied — using corpus mean %.2f → contamination=%.4f",
            mean_var, contamination,
        )

    params = {**IF_PARAMS, "contamination": contamination}
    logger.info("Training IsolationForest | contamination=%.4f | params: %s", contamination, params)
    model = IsolationForest(**params)
    model.fit(X)

    # Compute raw scores on training set to calibrate scaler
    raw_scores = model.score_samples(X)          # range: typically [-0.8, 0.0]
    # IF: more negative = more anomalous → invert so higher = riskier
    inverted_scores = -raw_scores                 # range: [0.0, ~0.8]

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(inverted_scores.reshape(-1, 1))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info("Model saved to %s", model_path)
    logger.info("Scaler saved to %s", scaler_path)

    return model, scaler


# ---------------------------------------------------------------------------
# Model loading (with lazy singleton cache)
# ---------------------------------------------------------------------------
_model_cache: Optional[IsolationForest] = None
_scaler_cache: Optional[MinMaxScaler] = None


def load_model(
    model_path: Path = IF_MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    retrain_if_missing: bool = True,
) -> tuple[IsolationForest, MinMaxScaler]:
    """
    Load the trained IsolationForest and scaler from disk.
    If artefacts are missing and ``retrain_if_missing`` is True, trains a
    fresh model on synthetic data first.

    Returns a cached singleton—safe to call on every request.
    """
    global _model_cache, _scaler_cache

    if _model_cache is not None and _scaler_cache is not None:
        return _model_cache, _scaler_cache

    if not model_path.exists() or not scaler_path.exists():
        if retrain_if_missing:
            logger.warning(
                "Model artefacts not found at %s — auto-training on synthetic data.",
                MODEL_DIR,
            )
            _model_cache, _scaler_cache = train_and_save_model(
                model_path=model_path, scaler_path=scaler_path
            )
        else:
            raise FileNotFoundError(
                f"Model artefacts missing: {model_path}, {scaler_path}. "
                "Run train_and_save_model() first."
            )
    else:
        _model_cache = joblib.load(model_path)
        _scaler_cache = joblib.load(scaler_path)
        logger.info("Loaded model from %s", model_path)

    return _model_cache, _scaler_cache


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------
def score_segment(
    feature_row: pd.DataFrame,
    model: Optional[IsolationForest] = None,
    scaler: Optional[MinMaxScaler] = None,
) -> tuple[float, float, tuple[float, float]]:
    """
    Run the IsolationForest on a single-row feature DataFrame and return:
        (raw_if_score, normalised_prob 0–1, (ci_lower, ci_upper))

    Parameters
    ----------
    feature_row : pd.DataFrame
        Single-row frame with exactly FEATURE_COLS columns (from build_feature_vector).
    model : IsolationForest, optional
        If None, the cached model is loaded automatically.
    scaler : MinMaxScaler, optional
        If None, the cached scaler is loaded automatically.

    Returns
    -------
    isolation_forest_raw_score : float
        Raw decision function score (negative = anomalous).
    normalised_risk_probability : float
        [0.0, 1.0] risk probability.
    confidence_interval : (float, float)
        95 % credible interval around the normalised probability.
    """
    if model is None or scaler is None:
        model, scaler = load_model()

    X = feature_row[FEATURE_COLS].values
    raw_score: float = float(model.score_samples(X)[0])

    inverted = -raw_score  # higher = more anomalous
    norm_prob: float = float(
        np.clip(scaler.transform([[inverted]])[0][0], 0.0, 1.0)
    )

    # Confidence interval: widen in the uncertain mid-zone
    mid_zone = 1.0 - abs(norm_prob - 0.5) * 2.0   # 0 at extremes, 1 at 0.5
    half_w = _CI_BASE_HALF_WIDTH + mid_zone * _CI_MID_ZONE_BOOST
    ci_lower = float(np.clip(norm_prob - half_w, 0.0, 1.0))
    ci_upper = float(np.clip(norm_prob + half_w, 0.0, 1.0))

    return raw_score, norm_prob, (ci_lower, ci_upper)


# ---------------------------------------------------------------------------
# Chennai Floods 2023 — model validation
# ---------------------------------------------------------------------------
def validate_chennai_scenario(verbose: bool = True) -> dict:
    """
    Run the model against the synthetic Chennai Floods timeline and confirm
    it flags disruption 4–6 hours before the historical closure timestamp.

    Returns a dict with per-timestep scores and a pass/fail verdict.
    """
    model, scaler = load_model()
    timeline = generate_chennai_flood_timeline()

    # Build dummy segment for NH48
    segment = {
        "segment_id":                "seg-nh48-chennai-flood-demo",
        "nh_identifier":             "NH48",
        "start_node_latlon":         [13.0827, 80.2707],
        "end_node_latlon":           [12.9716, 79.9592],
        "base_distance_km":          62.0,
        "historical_delay_variance": 18.5,
    }

    results = []
    first_flag_h: Optional[int] = None

    for step in timeline:
        # Strip internal metadata keys before building LiveSignals
        signals_kwargs = {
            k: v for k, v in step.items() if not k.startswith("_")
        }
        signals = LiveSignals(**signals_kwargs)
        features = build_feature_vector(segment, signals)
        raw, prob, ci = score_segment(features, model, scaler)

        flagged = prob >= 0.6   # Orange threshold
        h = step["_hours_before_closure"]

        if flagged and first_flag_h is None:
            first_flag_h = h

        results.append({
            "hours_before_closure": h,
            "observation_utc":      step["observation_utc"].isoformat(),
            "raw_score":            round(raw, 6),
            "risk_probability":     round(prob, 4),
            "ci":                   (round(ci[0], 4), round(ci[1], 4)),
            "flagged":              flagged,
        })

        if verbose:
            flag_str = "🔴 FLAG" if flagged else "  ✅  "
            logger.info(
                "[Chennai Validation] T%+3dh  prob=%.3f  %s",
                -h, prob, flag_str
            )

    pass_condition = first_flag_h is not None and 4 <= first_flag_h <= 6
    verdict = "PASS ✅" if pass_condition else "FAIL ❌"
    logger.info(
        "Chennai Demo Validation: %s  (first flag at T-%sh before closure)",
        verdict,
        first_flag_h,
    )

    return {
        "verdict": verdict,
        "first_flag_hours_before_closure": first_flag_h,
        "pass_condition_4_to_6h": pass_condition,
        "timeline": results,
    }


# ===========================================================================
# TIER 2 — PROPHET TIME-SERIES FORECASTING
# ===========================================================================
# Prophet is imported lazily inside functions to avoid a hard startup
# dependency; the app still starts even if Prophet is not installed in a
# lightweight deployment (Tier-1 IsolationForest will still function).
# ===========================================================================

from dataclasses import dataclass, field as dc_field  # noqa: E402 (post-block import)


# ---------------------------------------------------------------------------
# Prophet forecast result container
# ---------------------------------------------------------------------------
@dataclass
class ProphetForecastResult:
    """
    Output of a single Prophet forecast run for one highway segment.

    Fields
    ------
    segment_id : str
        The segment this forecast belongs to.
    forecast_generated_utc : str
        ISO-8601 UTC timestamp when forecast was computed.
    horizon_hours : list[int]
        The requested forecast horizons, e.g. [6, 12, 24].
    delay_probability_at_horizon : dict[int, float]
        Maps each horizon (hours) → predicted delay-disruption probability
        in [0.0, 1.0]. Derived by sigmoiding the Prophet yhat output.
    yhat_at_horizon : dict[int, float]
        Raw Prophet ``yhat`` (expected delay minutes) at each horizon.
    yhat_lower_at_horizon : dict[int, float]
        Prophet 95 % lower bound at each horizon.
    yhat_upper_at_horizon : dict[int, float]
        Prophet 95 % upper bound at each horizon.
    rainfall_regressor_weight : float
        The fitted beta coefficient for the rainfall_mm_6h regressor,
        indicating how strongly rainfall drives delay on this segment.
    seasonality_components : dict[str, float]
        Seasonality amplitude at the forecast horizon for each named
        component (indian_peak_hours, indian_festival, indian_weekend).
    """
    segment_id: str
    forecast_generated_utc: str
    horizon_hours: list[int]
    delay_probability_at_horizon: dict[int, float] = dc_field(default_factory=dict)
    yhat_at_horizon: dict[int, float] = dc_field(default_factory=dict)
    yhat_lower_at_horizon: dict[int, float] = dc_field(default_factory=dict)
    yhat_upper_at_horizon: dict[int, float] = dc_field(default_factory=dict)
    rainfall_regressor_weight: float = 0.0
    seasonality_components: dict[str, float] = dc_field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "segment_id":                   self.segment_id,
            "forecast_generated_utc":       self.forecast_generated_utc,
            "horizon_hours":                self.horizon_hours,
            "delay_probability_at_horizon": self.delay_probability_at_horizon,
            "yhat_at_horizon":              self.yhat_at_horizon,
            "yhat_lower_at_horizon":        self.yhat_lower_at_horizon,
            "yhat_upper_at_horizon":        self.yhat_upper_at_horizon,
            "rainfall_regressor_weight":    round(self.rainfall_regressor_weight, 6),
            "seasonality_components":       self.seasonality_components,
        }


# ---------------------------------------------------------------------------
# India-specific seasonality Fourier specifications
# ---------------------------------------------------------------------------
class IndianSeasonality:
    """
    Custom Prophet seasonality definitions modelling transport demand
    patterns specific to Indian national highways.

    All periods are in days (Prophet's native unit).
    Fourier order controls the number of sin/cos harmonics — higher order
    captures sharper intra-period peaks at the cost of potential overfitting.
    """

    # ------------------------------------------------------------------
    # 1. Indian peak transport hours
    #    Freight trucks peak 22:00–05:00 (avoid daytime restrictions).
    #    Passenger vehicles peak 07:00–10:00 and 17:00–21:00.
    #    Net effect: intra-day congestion / delay spikes at those windows.
    #    Period  = 1 day (24 h cycle)
    #    Fourier = 6  — captures both AM + PM peaks and the overnight ramp
    # ------------------------------------------------------------------
    PEAK_HOURS = dict(
        name="indian_peak_hours",
        period=1.0,
        fourier_order=6,
        prior_scale=5.0,   # moderately strong; peak hours are predictable
        mode="additive",
    )

    # ------------------------------------------------------------------
    # 2. Indian festival cluster seasonality
    #    Major festival clusters that cause sustained NH volume surges:
    #      - Diwali / Dussehra  (Oct–Nov, ~3 weeks)
    #      - Pongal / Makar Sankranti (mid-Jan)
    #      - Holi (Mar)
    #      - Eid-ul-Fitr (variable, ~Apr–May)
    #    Approximate annual repeat at ~365 / 4 ≈ 91.25-day sub-cycles.
    #    Fourier = 3 — broad cluster shape, not sharp spike
    # ------------------------------------------------------------------
    FESTIVAL = dict(
        name="indian_festival",
        period=91.3125,    # quarterly recurrence approximation
        fourier_order=3,
        prior_scale=8.0,   # stronger prior — festivals have large effects
        mode="additive",
    )

    # ------------------------------------------------------------------
    # 3. Indian weekend pattern
    #    India's commercial freight peaks on Fri–Sat (in anticipation of
    #    Sunday delivery deadlines).  Weekly 7-day period.
    #    Fourier = 3 — weekly rhythm with Fri/Sat asymmetry
    # ------------------------------------------------------------------
    WEEKEND = dict(
        name="indian_weekend",
        period=7.0,
        fourier_order=3,
        prior_scale=4.0,
        mode="additive",
    )

    ALL: list[dict] = [PEAK_HOURS, FESTIVAL, WEEKEND]


# ---------------------------------------------------------------------------
# Synthetic Prophet training series builder
# ---------------------------------------------------------------------------
def build_prophet_training_series(
    segment: dict,
    history_days: int = 90,
    observations_per_day: int = 24,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Build a synthetic hourly delay time-series for a segment that can be
    used to fit a Prophet model when no real historical DB data is available
    (e.g. during development, the Chennai demo, or cold-start).

    In production this function should be replaced (or supplemented) by a
    query to ``route_observations`` in Supabase returning columns:
        ds             : datetime (UTC, hourly)
        y              : float    (delay minutes)
        rainfall_mm_6h : float    (6-hour cumulative rainfall at observation)

    The synthetic series encodes:
      - A long-term rising trend (monsoon build-up over 90 days)
      - Intra-day peaks at 08:00 and 19:00 IST
      - Weekly Friday freight surge
      - Quarterly festival spikes (Diwali window embedded in the series)
      - rainfall_mm_6h as an additive regressor correlated with delay
      - Gaussian noise calibrated to the segment's historical_delay_variance

    Parameters
    ----------
    segment : dict
        Route segment dict (must contain ``historical_delay_variance``,
        ``base_distance_km``, ``nh_identifier``).
    history_days : int
        Number of days of hourly history to generate (default 90).
    observations_per_day : int
        Observations per day (default 24, i.e. hourly).
    rng_seed : int
        NumPy RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: ``ds`` (datetime64[ns] UTC), ``y`` (delay minutes),
        ``rainfall_mm_6h`` (float).
    """
    rng = np.random.default_rng(rng_seed)
    n = history_days * observations_per_day
    hist_var = max(float(segment.get("historical_delay_variance", 10.0)), 0.1)
    noise_std = float(np.sqrt(hist_var))

    # Base timestamps — hourly, ending at current UTC
    end_ts = pd.Timestamp.utcnow().floor("h")
    timestamps = pd.date_range(end=end_ts, periods=n, freq="h")

    t = np.linspace(0, history_days, n)            # fractional day index
    hour_of_day = np.array([ts.hour for ts in timestamps], dtype=float)
    day_of_week = np.array([ts.dayofweek for ts in timestamps], dtype=float)  # 0=Mon
    day_of_year = np.array([ts.dayofyear for ts in timestamps], dtype=float)

    # --- Trend: slow monsoon build-up over the period ---
    trend = 5.0 + t * 0.12

    # --- Intra-day peak seasonality (08:00 and 19:00 IST = 02:30 and 13:30 UTC) ---
    peak_am_utc = 2.5   # 08:00 IST
    peak_pm_utc = 13.5  # 19:00 IST
    intraday = (
        6.0 * np.exp(-0.5 * ((hour_of_day - peak_am_utc) / 1.5) ** 2)
        + 8.0 * np.exp(-0.5 * ((hour_of_day - peak_pm_utc) / 2.0) ** 2)
    )

    # --- Weekly: Friday (dayofweek=4) freight surge ---
    weekly = np.where(day_of_week == 4, 5.0, 0.0)
    weekly += np.where(day_of_week == 5, 3.0, 0.0)   # Saturday also elevated

    # --- Quarterly festival cluster (Diwali Nov window ≈ day 300–320 of year) ---
    festival_peak = np.exp(
        -0.5 * ((day_of_year - 310) / 12.0) ** 2
    ) * 12.0

    # --- rainfall_mm_6h: synthetic — ramps up mid-series (monsoon simulation) ---
    rain_base = rng.uniform(0, 5, n)
    monsoon_ramp = np.clip(np.sin(np.pi * t / history_days), 0, 1)  # 0→1→0 bell
    rainfall_mm_6h = rain_base + monsoon_ramp * rng.uniform(30, 80, n)
    rainfall_mm_6h = np.clip(rainfall_mm_6h, 0.0, 350.0)

    # Rainfall contributes additively to delay
    rain_effect = rainfall_mm_6h * 0.08   # 0.08 min delay per mm (calibrated)

    # --- Combine + noise ---
    y = (
        trend
        + intraday
        + weekly
        + festival_peak
        + rain_effect
        + rng.normal(0, noise_std, n)
    )
    y = np.clip(y, 0.0, 600.0)   # delay capped at 10 hours

    return pd.DataFrame({
        "ds":             timestamps,
        "y":              y,
        "rainfall_mm_6h": rainfall_mm_6h,
    })


# ---------------------------------------------------------------------------
# Prophet model fitter
# ---------------------------------------------------------------------------
def fit_prophet_model(
    history_df: pd.DataFrame,
    segment_id: str,
    historical_delay_variance: float = 10.0,
    horizon_hours: list[int] | None = None,
) -> "prophet.Prophet":  # type: ignore[name-defined]
    """
    Fit a Prophet model on a segment's delay history with India-specific
    seasonalities and a rainfall_mm_6h additive regressor.

    Parameters
    ----------
    history_df : pd.DataFrame
        Columns required: ``ds`` (datetime), ``y`` (delay minutes),
        ``rainfall_mm_6h`` (float).
    segment_id : str
        Used for logging context only.
    historical_delay_variance : float
        Drives the changepoint_prior_scale: more volatile segments allow
        the trend to change more freely.
    horizon_hours : list[int], optional
        Not used during fitting; carried for logging clarity.
        Defaults to [6, 12, 24].

    Returns
    -------
    prophet.Prophet
        A fitted Prophet model with the custom seasonalities attached.

    Raises
    ------
    ImportError
        If ``prophet`` is not installed.
    ValueError
        If ``history_df`` is missing required columns or has < 2 rows.
    """
    _horizon_hours = horizon_hours or [6, 12, 24]

    try:
        from prophet import Prophet  # lazy import
    except ImportError as exc:
        raise ImportError(
            "prophet is not installed. Run: pip install prophet"
        ) from exc

    # --- Validate input ---
    required_cols = {"ds", "y", "rainfall_mm_6h"}
    missing_cols = required_cols - set(history_df.columns)
    if missing_cols:
        raise ValueError(
            f"history_df is missing columns: {missing_cols}. "
            "Expected: ds (datetime), y (delay minutes), rainfall_mm_6h (float)."
        )
    if len(history_df) < 2:
        raise ValueError(
            f"history_df has {len(history_df)} row(s); Prophet requires at least 2."
        )

    # --- Changepoint prior: more volatile segments get more flexible trends ---
    # Normalised from [1, 80] variance range → [0.05, 0.5] changepoint scale
    _cp_low, _cp_high = 0.05, 0.50
    t_var = float(np.clip(
        (historical_delay_variance - _VAR_LOW) / (_VAR_HIGH - _VAR_LOW), 0.0, 1.0
    ))
    changepoint_prior_scale = _cp_low + t_var * (_cp_high - _cp_low)

    logger.info(
        "[Prophet] Fitting segment=%s | rows=%d | horizons=%s | "
        "changepoint_prior=%.3f | delay_var=%.2f",
        segment_id, len(history_df), _horizon_hours,
        changepoint_prior_scale, historical_delay_variance,
    )

    # --- Build model ---
    model = Prophet(
        growth="linear",
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=10.0,   # default; individual components override
        holidays_prior_scale=10.0,
        seasonality_mode="additive",    # rainfall and seasonalities sum linearly
        interval_width=0.95,
        daily_seasonality=False,        # replaced by indian_peak_hours
        weekly_seasonality=False,       # replaced by indian_weekend
        yearly_seasonality=False,       # replaced by indian_festival
    )

    # --- Attach India-specific seasonalities ---
    for spec in IndianSeasonality.ALL:
        model.add_seasonality(**spec)
        logger.debug("[Prophet] Added seasonality: %s", spec["name"])

    # --- Rainfall as additive external regressor ---
    # Standardise in-place on a copy to keep history_df immutable
    train_df = history_df[["ds", "y", "rainfall_mm_6h"]].copy()
    rain_mean = train_df["rainfall_mm_6h"].mean()
    rain_std = max(train_df["rainfall_mm_6h"].std(), 1e-6)
    train_df["rainfall_mm_6h_std"] = (
        (train_df["rainfall_mm_6h"] - rain_mean) / rain_std
    )
    model.add_regressor(
        "rainfall_mm_6h_std",
        prior_scale=10.0,   # allow rainfall to have strong influence
        standardize=False,  # already standardised above
        mode="additive",
    )

    # Rename column to match regressor name Prophet expects
    train_df = train_df.rename(columns={"rainfall_mm_6h": "_rain_raw"})
    train_df = train_df.rename(columns={"rainfall_mm_6h_std": "rainfall_mm_6h_std"})
    train_df["ds"] = pd.to_datetime(train_df["ds"]).dt.tz_localize(None)  # Prophet requires tz-naive

    model.fit(train_df[["ds", "y", "rainfall_mm_6h_std"]])

    # Stash normalisation params on the model for use during forecast
    model._rain_mean = rain_mean   # type: ignore[attr-defined]
    model._rain_std = rain_std     # type: ignore[attr-defined]

    logger.info("[Prophet] Fit complete for segment=%s", segment_id)
    return model


# ---------------------------------------------------------------------------
# Forecast function — produces 6/12/24-hour window predictions
# ---------------------------------------------------------------------------
def forecast_delay_probability(
    model: "prophet.Prophet",  # type: ignore[name-defined]
    segment_id: str,
    horizon_hours: list[int] | None = None,
    future_rainfall_mm_6h: float = 0.0,
) -> ProphetForecastResult:
    """
    Use a fitted Prophet model to forecast delay probability at each
    requested horizon (6 h / 12 h / 24 h rolling window).

    The raw ``yhat`` (expected delay in minutes) is mapped to a
    disruption probability via a sigmoid function centred at 30 minutes
    (configurable through ``_SIGMOID_CENTRE`` below) with a softness
    controlled by ``_SIGMOID_SCALE``.

    Parameters
    ----------
    model : Prophet
        A Prophet model returned by ``fit_prophet_model()``.
    segment_id : str
        Used to populate the result container.
    horizon_hours : list[int], optional
        Horizons to forecast (default [6, 12, 24]).
    future_rainfall_mm_6h : float
        Expected 6-hour cumulative rainfall for the forecast period.
        Used to populate the rainfall regressor in the ``make_future_dataframe``.
        Defaults to 0.0 (dry weather assumption).

    Returns
    -------
    ProphetForecastResult
    """
    from datetime import timezone as _tz

    _horizon_hours = sorted(set(horizon_hours or [6, 12, 24]))
    max_horizon_h = max(_horizon_hours)

    # --- Build future dataframe covering max horizon ---
    # make_future_dataframe works in day increments; use hourly periods
    future = model.make_future_dataframe(
        periods=max_horizon_h,
        freq="h",
        include_history=False,
    )
    future["ds"] = pd.to_datetime(future["ds"]).dt.tz_localize(None)

    # Populate rainfall regressor for the forecast window
    rain_mean: float = getattr(model, "_rain_mean", 0.0)
    rain_std: float = getattr(model, "_rain_std", 1.0)
    rain_std = max(rain_std, 1e-6)
    future["rainfall_mm_6h_std"] = (future_rainfall_mm_6h - rain_mean) / rain_std

    forecast = model.predict(future)

    # --- Sigmoid mapping: delay_minutes → probability ---
    # Sigmoid centred at 30 min: P=0.5 at 30 min, P→1 as delay→∞
    # Scale 15 min: P reaches ~0.88 at 60 min, ~0.27 at 10 min
    _SIGMOID_CENTRE = 30.0   # minutes at which P = 0.5
    _SIGMOID_SCALE = 15.0    # controls steepness

    def _delay_to_prob(delay_min: float) -> float:
        return float(1.0 / (1.0 + np.exp(-(delay_min - _SIGMOID_CENTRE) / _SIGMOID_SCALE)))

    # --- Extract per-horizon values ---
    delay_prob: dict[int, float] = {}
    yhat_h: dict[int, float] = {}
    yhat_lower_h: dict[int, float] = {}
    yhat_upper_h: dict[int, float] = {}

    for h in _horizon_hours:
        # forecast is ordered chronologically; row index h-1 = h hours ahead
        idx = min(h - 1, len(forecast) - 1)
        row = forecast.iloc[idx]
        yhat_val = float(np.clip(row["yhat"], 0.0, 600.0))
        yhat_h[h] = round(yhat_val, 2)
        yhat_lower_h[h] = round(float(np.clip(row["yhat_lower"], 0.0, 600.0)), 2)
        yhat_upper_h[h] = round(float(np.clip(row["yhat_upper"], 0.0, 600.0)), 2)
        delay_prob[h] = round(_delay_to_prob(yhat_val), 6)

    # --- Rainfall regressor beta coefficient ---
    rain_weight = 0.0
    if hasattr(model, "params") and "beta" in model.params:
        # Prophet stores regressor coefficients in model.params["beta"]
        # The rainfall regressor index is determined by insertion order
        try:
            reg_names = [r["name"] for r in model.extra_regressors.values()]  # type: ignore[attr-defined]
            rain_idx = reg_names.index("rainfall_mm_6h_std")
            rain_weight = float(np.mean(model.params["beta"][:, rain_idx]))
        except (ValueError, KeyError, IndexError, AttributeError):
            pass

    # --- Seasonality component amplitudes at the nearest horizon ---
    seasonality_components: dict[str, float] = {}
    seasonality_cols = [
        "additive_terms",
        "indian_peak_hours",
        "indian_festival",
        "indian_weekend",
    ]
    ref_idx = min(max_horizon_h - 1, len(forecast) - 1)
    for col in seasonality_cols:
        if col in forecast.columns:
            seasonality_components[col] = round(float(forecast[col].iloc[ref_idx]), 4)

    forecast_generated_utc = datetime.now(timezone.utc).isoformat()

    result = ProphetForecastResult(
        segment_id=segment_id,
        forecast_generated_utc=forecast_generated_utc,
        horizon_hours=_horizon_hours,
        delay_probability_at_horizon=delay_prob,
        yhat_at_horizon=yhat_h,
        yhat_lower_at_horizon=yhat_lower_h,
        yhat_upper_at_horizon=yhat_upper_h,
        rainfall_regressor_weight=rain_weight,
        seasonality_components=seasonality_components,
    )

    logger.info(
        "[Prophet] Forecast | segment=%s | horizons=%s | probs=%s",
        segment_id,
        _horizon_hours,
        {h: f"{p:.3f}" for h, p in delay_prob.items()},
    )
    return result


# ---------------------------------------------------------------------------
# Convenience wrapper: fit (or retrieve cached) + forecast in one call
# ---------------------------------------------------------------------------

# In-process LRU cache keyed on segment_id.
# Stores (prophet_model, history_fingerprint) tuples.
# Size 32 covers typical demo scenarios (32 unique NH segments).
_PROPHET_CACHE: dict[str, tuple] = {}
_PROPHET_CACHE_MAX = 32


def get_or_fit_prophet(
    segment: dict,
    history_df: Optional[pd.DataFrame] = None,
    horizon_hours: list[int] | None = None,
    future_rainfall_mm_6h: float = 0.0,
    force_refit: bool = False,
) -> ProphetForecastResult:
    """
    High-level entry point: fits a Prophet model for the segment (or
    retrieves it from the in-process cache) and returns a forecast.

    Called by ``routes/disruptions.py`` when the /predict endpoint includes
    ``include_forecast=true`` in the query parameters.

    Parameters
    ----------
    segment : dict
        Route segment dict (must include ``segment_id``,
        ``historical_delay_variance``, ``base_distance_km``).
    history_df : pd.DataFrame, optional
        Pre-fetched hourly delay history.  If None, a synthetic series is
        generated via ``build_prophet_training_series()`` (demo / cold-start).
    horizon_hours : list[int], optional
        Forecast horizons in hours (default [6, 12, 24]).
    future_rainfall_mm_6h : float
        Expected rainfall for the forecast window (used as regressor).
    force_refit : bool
        If True, bypass the cache and refit even if a model exists.

    Returns
    -------
    ProphetForecastResult
    """
    _horizon_hours = horizon_hours or [6, 12, 24]
    seg_id: str = segment["segment_id"]
    hist_var = float(segment.get("historical_delay_variance", 10.0))

    # --- Cache lookup ---
    if not force_refit and seg_id in _PROPHET_CACHE:
        cached_model, _ = _PROPHET_CACHE[seg_id]
        logger.debug("[Prophet] Cache hit for segment=%s", seg_id)
    else:
        # Build history if not provided
        if history_df is None:
            logger.info(
                "[Prophet] No history_df supplied for segment=%s — "
                "generating synthetic 90-day series.", seg_id,
            )
            history_df = build_prophet_training_series(segment)

        cached_model = fit_prophet_model(
            history_df=history_df,
            segment_id=seg_id,
            historical_delay_variance=hist_var,
            horizon_hours=_horizon_hours,
        )

        # Evict oldest entry when cache is full
        if len(_PROPHET_CACHE) >= _PROPHET_CACHE_MAX:
            oldest_key = next(iter(_PROPHET_CACHE))
            del _PROPHET_CACHE[oldest_key]
            logger.debug("[Prophet] Cache evicted segment=%s (LRU)", oldest_key)

        _PROPHET_CACHE[seg_id] = (cached_model, id(history_df))
        logger.info("[Prophet] Cached model for segment=%s (cache size=%d)", seg_id, len(_PROPHET_CACHE))

    return forecast_delay_probability(
        model=cached_model,
        segment_id=seg_id,
        horizon_hours=_horizon_hours,
        future_rainfall_mm_6h=future_rainfall_mm_6h,
    )
