"""
feature_engineering.py — FairChain Disruption Engine
=====================================================
Transforms raw route segment data + live / simulated signals into the
feature vector consumed by the Isolation Forest disruption model.

Owned by: ML Lead (Disruption Engine)
DO NOT MODIFY outside the 4 ML-owned files.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature names — must stay in sync with disruption_model.py FEATURE_COLS
# ---------------------------------------------------------------------------
FEATURE_COLS: list[str] = [
    "rainfall_mm_1h",          # Precipitation in last 1 hour (mm)
    "rainfall_mm_6h",          # Precipitation in last 6 hours (mm)
    "velocity_kmh",            # Current average vehicle speed on segment (km/h)
    "velocity_deviation_pct",  # % deviation of velocity from historical μ
    "incident_count_2h",       # # of incidents reported on segment in last 2 h
    "visibility_km",           # Visibility in km (weather service)
    "water_level_m",           # River / drain water level (metres)
    "segment_load_factor",     # # active vehicles / segment capacity (0–∞)
    "delay_z_score",           # (current_delay – historical_μ) / historical_σ
    "temp_celsius",            # Ambient temperature (°C)
    "wind_speed_kmh",          # Wind speed km/h
    "is_night",                # 1 if observation between 21:00–06:00 else 0
    "distance_km",             # segment base distance (structural feature)
    "historical_delay_var",    # pre-computed variance from segment metadata
]

# ---------------------------------------------------------------------------
# Anomalous feature thresholds  (used to derive dominant_anomalous_features)
# ---------------------------------------------------------------------------
ANOMALY_THRESHOLDS: dict[str, tuple[str, float]] = {
    # feature_col -> (direction, threshold_for_anomaly)
    # direction: "above" | "below"
    "rainfall_mm_1h":         ("above",  30.0),
    "rainfall_mm_6h":         ("above",  80.0),
    "velocity_kmh":           ("below",  20.0),
    "velocity_deviation_pct": ("above",  40.0),
    "incident_count_2h":      ("above",   4.0),
    "visibility_km":          ("below",   0.5),
    "water_level_m":          ("above",   2.5),
    "segment_load_factor":    ("above",   1.8),
    "delay_z_score":          ("above",   2.5),
    "wind_speed_kmh":         ("above",  60.0),
}

# Human-readable alias map for dominant_anomalous_features output
FEATURE_ALIAS: dict[str, str] = {
    "rainfall_mm_1h":         "rainfall_spike",
    "rainfall_mm_6h":         "sustained_rainfall",
    "velocity_kmh":           "velocity_plunge",
    "velocity_deviation_pct": "speed_anomaly",
    "incident_count_2h":      "incident_surge",
    "visibility_km":          "low_visibility",
    "water_level_m":          "flooding_risk",
    "segment_load_factor":    "congestion_overload",
    "delay_z_score":          "delay_spike",
    "wind_speed_kmh":         "high_winds",
}


# ---------------------------------------------------------------------------
# Raw signal dataclass — the "live" payload fed to feature engineering
# ---------------------------------------------------------------------------
@dataclass
class LiveSignals:
    """
    Represents live / near-real-time sensor and weather signals for a
    specific route segment at a given timestamp.

    In production these values arrive from external APIs (weather services,
    ATMS traffic feeds).  For the demo simulation they are injected by the
    Chennai Floods replay utility below.
    """
    rainfall_mm_1h: float = 0.0
    rainfall_mm_6h: float = 0.0
    velocity_kmh: float = 60.0
    incident_count_2h: int = 0
    visibility_km: float = 10.0
    water_level_m: float = 0.5
    segment_load_factor: float = 0.6
    current_delay_min: float = 0.0         # used to compute delay_z_score
    historical_avg_delay_min: float = 5.0  # from segment metadata / DB
    temp_celsius: float = 28.0
    wind_speed_kmh: float = 15.0
    observation_utc: Optional[datetime] = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Core feature engineering function
# ---------------------------------------------------------------------------
def build_feature_vector(
    segment: dict,
    signals: LiveSignals,
) -> pd.DataFrame:
    """
    Combine route segment metadata with live signals to produce a single-row
    DataFrame of FEATURE_COLS ready for IsolationForest inference.

    Parameters
    ----------
    segment : dict
        Route segment object matching the agreed API schema:
        {
            "segment_id": str,
            "nh_identifier": str,
            "start_node_latlon": [float, float],
            "end_node_latlon": [float, float],
            "base_distance_km": float,
            "historical_delay_variance": float
        }
    signals : LiveSignals
        Live or simulated sensor / weather readings.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns == FEATURE_COLS (same order).
    """
    # --- Derived features ---
    obs_hour = (
        signals.observation_utc.hour
        if signals.observation_utc
        else datetime.now(timezone.utc).hour
    )
    is_night = int(obs_hour >= 21 or obs_hour < 6)

    hist_avg = max(signals.historical_avg_delay_min, 1e-6)
    hist_var = max(segment.get("historical_delay_variance", 1.0), 1e-6)
    hist_std = math.sqrt(hist_var)

    delay_z_score = (signals.current_delay_min - hist_avg) / hist_std

    # Velocity deviation from a "normal" 60 km/h baseline
    # (production: replace baseline with segment-level historical μ)
    normal_velocity = 60.0
    velocity_deviation_pct = (
        abs(normal_velocity - signals.velocity_kmh) / normal_velocity * 100.0
    )

    row = {
        "rainfall_mm_1h":         signals.rainfall_mm_1h,
        "rainfall_mm_6h":         signals.rainfall_mm_6h,
        "velocity_kmh":           signals.velocity_kmh,
        "velocity_deviation_pct": velocity_deviation_pct,
        "incident_count_2h":      float(signals.incident_count_2h),
        "visibility_km":          signals.visibility_km,
        "water_level_m":          signals.water_level_m,
        "segment_load_factor":    signals.segment_load_factor,
        "delay_z_score":          delay_z_score,
        "temp_celsius":           signals.temp_celsius,
        "wind_speed_kmh":         signals.wind_speed_kmh,
        "is_night":               float(is_night),
        "distance_km":            segment.get("base_distance_km", 0.0),
        "historical_delay_var":   hist_var,
    }

    return pd.DataFrame([row], columns=FEATURE_COLS)


# ---------------------------------------------------------------------------
# Dominant anomalous features detector
# ---------------------------------------------------------------------------
def detect_dominant_anomalous_features(feature_row: pd.DataFrame) -> list[str]:
    """
    Given a single-row feature DataFrame, return a list of human-readable
    aliases for features that exceed their anomaly thresholds.

    Returns at most 5 features, sorted by severity (most extreme first).
    """
    violations: list[tuple[str, float]] = []

    for col, (direction, threshold) in ANOMALY_THRESHOLDS.items():
        if col not in feature_row.columns:
            continue
        val = float(feature_row[col].iloc[0])
        if direction == "above" and val > threshold:
            severity = (val - threshold) / (threshold + 1e-9)
            violations.append((col, severity))
        elif direction == "below" and val < threshold:
            severity = (threshold - val) / (threshold + 1e-9)
            violations.append((col, severity))

    # Sort by severity descending, cap at 5
    violations.sort(key=lambda x: x[1], reverse=True)
    return [FEATURE_ALIAS.get(col, col) for col, _ in violations[:5]]


# ---------------------------------------------------------------------------
# Chennai Floods 2023 — Simulation Replay Utility
# -----------------------------------------------------------------------
# Historical facts used for simulation:
#   - Event date     : 2–4 November 2023
#   - Affected NH   : NH48 (Chennai–Bengaluru corridor)
#   - Actual closure: ~06:00 IST 3 Nov 2023
#   - Target        : model must flag disruption T-4 to T-6 hours before
#                     closure, i.e. ~00:00–02:00 IST 3 Nov 2023
# ---------------------------------------------------------------------------

def generate_chennai_flood_timeline(
    segment_id: str = "seg-nh48-chennai-flood-demo",
    steps_before_closure_h: int = 12,
    rng_seed: int = 2023,
) -> list[dict]:
    """
    Generate a synthetic time-series of LiveSignals dicts representing the
    build-up to the Nov 2023 Chennai Floods road closure on NH48.

    Returns a list of dicts (each compatible with LiveSignals(**d)) ordered
    chronologically from T-steps_before_closure_h to T+0 (closure).

    The disruption signal ramps up linearly over the last 6 hours before
    closure so the IsolationForest should flag it 4–6 h in advance.
    """
    rng = random.Random(rng_seed)
    # Closure timestamp in UTC (06:00 IST = 00:30 UTC)
    closure_utc = datetime(2023, 11, 3, 0, 30, 0, tzinfo=timezone.utc)

    timeline: list[dict] = []

    for h in range(steps_before_closure_h, -1, -1):
        # Hours before closure
        t = closure_utc - pd.Timedelta(hours=h)
        hours_to_closure = h

        # Disruption intensity: starts rising steeply 6 h before closure
        ramp = max(0.0, (6 - hours_to_closure) / 6.0)   # 0 → 1 as h → 0
        pre_ramp = max(0.0, (9 - hours_to_closure) / 9.0)  # subtle early signal

        # Rainfall escalates dramatically
        rainfall_1h = (
            2.0 + pre_ramp * 15 + ramp * 85
            + rng.gauss(0, 2.0)
        )
        rainfall_6h = (
            10.0 + pre_ramp * 40 + ramp * 260
            + rng.gauss(0, 5.0)
        )

        # Velocity drops as flooding begins
        velocity = max(
            2.0,
            60.0 - pre_ramp * 12 - ramp * 52
            + rng.gauss(0, 3.0),
        )

        # Water level at nearby Adyar River monitoring station
        water_level = min(
            4.5,
            0.6 + pre_ramp * 0.5 + ramp * 3.2
            + rng.gauss(0, 0.1),
        )

        # Visibility degrades with heavy rain
        visibility = max(
            0.05,
            10.0 - pre_ramp * 2 - ramp * 9.5
            + rng.gauss(0, 0.3),
        )

        # Incidents surge
        incident_count = int(max(0, ramp * 9 + rng.gauss(0, 1)))

        # Load factor rises as vehicles slow/stop
        load_factor = min(3.5, 0.5 + pre_ramp * 0.4 + ramp * 2.4)

        # Current delay piles up
        current_delay = max(0.0, 3.0 + pre_ramp * 20 + ramp * 180)

        # Temperature drops slightly during heavy rain
        temp = 29.0 - ramp * 4.0 + rng.gauss(0, 0.5)

        # Wind speed rises
        wind_speed = 12.0 + pre_ramp * 8 + ramp * 40 + rng.gauss(0, 2)

        timeline.append({
            "rainfall_mm_1h":           round(max(0.0, rainfall_1h), 2),
            "rainfall_mm_6h":           round(max(0.0, rainfall_6h), 2),
            "velocity_kmh":             round(velocity, 2),
            "incident_count_2h":        incident_count,
            "visibility_km":            round(max(0.05, visibility), 2),
            "water_level_m":            round(water_level, 3),
            "segment_load_factor":      round(load_factor, 3),
            "current_delay_min":        round(current_delay, 2),
            "historical_avg_delay_min": 8.0,   # historical baseline for NH48
            "temp_celsius":             round(temp, 1),
            "wind_speed_kmh":           round(wind_speed, 1),
            "observation_utc":          t,
            # metadata for replay endpoint
            "_hours_before_closure":    h,
            "_segment_id":              segment_id,
            "_event":                   "chennai_floods_nov_2023",
        })

    return timeline
