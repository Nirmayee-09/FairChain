"""
disruption_model.py — FairChain Disruption Engine
==================================================
Defines, trains, and persists the Isolation Forest anomaly detection model
used to score supply-chain route segments for disruption risk.

Architecture
------------
1. IsolationForest (sklearn) — primary anomaly scorer
2. MinMaxScaler — normalises raw IF scores to [0.0, 1.0] risk probability
3. Prophet trend baseline (optional, decoupled) — used offline for
   historical delay anomaly context during training data generation

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
