"""
predict_pipeline.py — FairChain Disruption Engine
==================================================
End-to-end prediction pipeline.  Accepts a raw route segment dict + optional
pre-built LiveSignals, orchestrates feature engineering → IsolationForest
scoring → output assembly into the exact API response schema.

This is the single entry point called by routes/disruptions.py.

Owned by: ML Lead (Disruption Engine)
DO NOT MODIFY outside the 4 ML-owned files.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from data.feature_engineering import (
    LiveSignals,
    build_feature_vector,
    detect_dominant_anomalous_features,
)
from models.disruption_model import load_model, score_segment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema dataclass (mirrors the agreed JSON contract exactly)
# ---------------------------------------------------------------------------
class PredictionResult:
    """
    Structured prediction output matching the /predict endpoint schema:

    {
        "segment_id":                    str (uuid),
        "current_timestamp_utc":         str (ISO-8601),
        "isolation_forest_raw_score":    float,
        "normalized_risk_probability":   float  [0.0, 1.0],
        "dominant_anomalous_features":   list[str],
        "model_confidence_interval":     [float, float]
    }
    """

    __slots__ = (
        "segment_id",
        "current_timestamp_utc",
        "isolation_forest_raw_score",
        "normalized_risk_probability",
        "dominant_anomalous_features",
        "model_confidence_interval",
    )

    def __init__(
        self,
        segment_id: str,
        current_timestamp_utc: str,
        isolation_forest_raw_score: float,
        normalized_risk_probability: float,
        dominant_anomalous_features: list[str],
        model_confidence_interval: list[float],
    ) -> None:
        self.segment_id = segment_id
        self.current_timestamp_utc = current_timestamp_utc
        self.isolation_forest_raw_score = isolation_forest_raw_score
        self.normalized_risk_probability = normalized_risk_probability
        self.dominant_anomalous_features = dominant_anomalous_features
        self.model_confidence_interval = model_confidence_interval

    def to_dict(self) -> dict:
        """Serialise to the agreed JSON schema dict."""
        return {
            "segment_id":                   self.segment_id,
            "current_timestamp_utc":        self.current_timestamp_utc,
            "isolation_forest_raw_score":   round(self.isolation_forest_raw_score, 6),
            "normalized_risk_probability":  round(self.normalized_risk_probability, 6),
            "dominant_anomalous_features":  self.dominant_anomalous_features,
            "model_confidence_interval":    [
                round(self.model_confidence_interval[0], 6),
                round(self.model_confidence_interval[1], 6),
            ],
        }


# ---------------------------------------------------------------------------
# Risk tier helper (for internal logging / Supabase enrichment)
# ---------------------------------------------------------------------------
def risk_tier(prob: float) -> str:
    """Map normalized risk probability to the agreed frontend colour tier."""
    if prob < 0.3:
        return "GREEN"
    elif prob < 0.6:
        return "YELLOW"
    elif prob < 0.8:
        return "ORANGE"
    return "RED"


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------
def run_prediction(
    segment: dict,
    signals: Optional[LiveSignals] = None,
) -> PredictionResult:
    """
    Full prediction pipeline: feature engineering → anomaly scoring → output.

    Parameters
    ----------
    segment : dict
        Route segment object from the API request body:
        {
            "segment_id":              str,
            "nh_identifier":           str,
            "start_node_latlon":       [float, float],
            "end_node_latlon":         [float, float],
            "base_distance_km":        float,
            "historical_delay_variance": float
        }

    signals : LiveSignals, optional
        Live or simulated sensor + weather readings.
        If None, a default (low-risk) LiveSignals object is used — useful
        for health-check / smoke-test calls.

    Returns
    -------
    PredictionResult
        Fully populated prediction matching the agreed API schema.

    Raises
    ------
    ValueError
        If ``segment`` is missing required keys.
    RuntimeError
        If model loading fails and auto-training is disabled.
    """
    # --- Validate required segment keys ---
    required_keys = {
        "segment_id",
        "nh_identifier",
        "start_node_latlon",
        "end_node_latlon",
        "base_distance_km",
        "historical_delay_variance",
    }
    missing = required_keys - set(segment.keys())
    if missing:
        raise ValueError(f"Segment dict missing required keys: {missing}")

    # Use default (benign) signals if none provided
    if signals is None:
        signals = LiveSignals()
        logger.debug(
            "No LiveSignals provided for segment %s — using default (low-risk) signals.",
            segment["segment_id"],
        )

    # --- Feature engineering ---
    feature_row = build_feature_vector(segment, signals)

    # --- Model scoring ---
    model, scaler = load_model()
    raw_score, norm_prob, ci_tuple = score_segment(feature_row, model, scaler)

    # --- Dominant anomalous features ---
    anomalous_features = detect_dominant_anomalous_features(feature_row)

    # --- Assemble output ---
    timestamp_utc = (
        signals.observation_utc
        if signals.observation_utc
        else datetime.now(timezone.utc)
    ).isoformat()

    result = PredictionResult(
        segment_id=segment["segment_id"],
        current_timestamp_utc=timestamp_utc,
        isolation_forest_raw_score=raw_score,
        normalized_risk_probability=norm_prob,
        dominant_anomalous_features=anomalous_features,
        model_confidence_interval=list(ci_tuple),
    )

    logger.info(
        "Prediction | segment=%s  nh=%s  prob=%.4f  tier=%s  anomalies=%s",
        segment["segment_id"],
        segment["nh_identifier"],
        norm_prob,
        risk_tier(norm_prob),
        anomalous_features,
    )

    return result


# ---------------------------------------------------------------------------
# Batch pipeline  (useful for replay / bulk Supabase writes)
# ---------------------------------------------------------------------------
def run_batch_predictions(
    segments: list[dict],
    signals_list: Optional[list[LiveSignals]] = None,
) -> list[PredictionResult]:
    """
    Run run_prediction() over a list of segments.

    Parameters
    ----------
    segments : list[dict]
        List of segment dicts (each must conform to the segment schema).
    signals_list : list[LiveSignals], optional
        Parallel list of LiveSignals objects.  If None or shorter than
        ``segments``, remaining segments get default signals.

    Returns
    -------
    list[PredictionResult]
        One result per segment, in the same order.
    """
    # Pre-load model once outside the loop
    load_model()

    results: list[PredictionResult] = []
    for i, seg in enumerate(segments):
        sig = (
            signals_list[i]
            if signals_list and i < len(signals_list)
            else None
        )
        try:
            results.append(run_prediction(seg, sig))
        except Exception as exc:
            logger.error(
                "Batch prediction failed for segment %s: %s",
                seg.get("segment_id", "?"),
                exc,
            )
            raise

    return results
