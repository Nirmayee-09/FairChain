"""
disruptions.py — FairChain Disruption Engine Routes
====================================================
FastAPI router exposing the ML disruption prediction endpoints.

Endpoints
---------
POST /predict              — score a single route segment
POST /predict/batch        — score multiple segments in one call
GET  /demo/chennai-replay  — stream the Nov 2023 Chennai Floods simulation
GET  /model/health         — model load / readiness probe

Owned by: ML Lead (Disruption Engine)
DO NOT MODIFY outside the 4 ML-owned files.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from data.feature_engineering import (
    LiveSignals,
    generate_chennai_flood_timeline,
)
from models.disruption_model import load_model, validate_chennai_scenario
from models.predict_pipeline import PredictionResult, risk_tier, run_batch_predictions, run_prediction

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/disruptions", tags=["disruptions"])


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class SegmentRequest(BaseModel):
    """Route segment object — matches the agreed input schema exactly."""
    segment_id: str = Field(..., description="UUID of the route segment")
    nh_identifier: str = Field(..., example="NH48")
    start_node_latlon: list[float] = Field(..., min_length=2, max_length=2)
    end_node_latlon: list[float] = Field(..., min_length=2, max_length=2)
    base_distance_km: float = Field(..., gt=0)
    historical_delay_variance: float = Field(..., ge=0)

    @field_validator("start_node_latlon", "end_node_latlon")
    @classmethod
    def validate_latlon(cls, v: list[float]) -> list[float]:
        lat, lon = v
        if not (-90 <= lat <= 90):
            raise ValueError(f"Latitude {lat} out of range [-90, 90]")
        if not (-180 <= lon <= 180):
            raise ValueError(f"Longitude {lon} out of range [-180, 180]")
        return v


class LiveSignalsRequest(BaseModel):
    """Optional live / simulated signal payload."""
    rainfall_mm_1h: float = Field(0.0, ge=0)
    rainfall_mm_6h: float = Field(0.0, ge=0)
    velocity_kmh: float = Field(60.0, ge=0)
    incident_count_2h: int = Field(0, ge=0)
    visibility_km: float = Field(10.0, ge=0)
    water_level_m: float = Field(0.5, ge=0)
    segment_load_factor: float = Field(0.6, ge=0)
    current_delay_min: float = Field(0.0, ge=0)
    historical_avg_delay_min: float = Field(5.0, ge=0)
    temp_celsius: float = Field(28.0)
    wind_speed_kmh: float = Field(15.0, ge=0)
    observation_utc: Optional[str] = Field(
        None, description="ISO-8601 UTC timestamp; defaults to server time"
    )


class PredictRequest(BaseModel):
    """Full prediction request body."""
    segment: SegmentRequest
    signals: Optional[LiveSignalsRequest] = None


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""
    items: list[PredictRequest] = Field(..., min_length=1, max_length=50)


class PredictionResponse(BaseModel):
    """Prediction response — exact schema agreed with frontend team."""
    segment_id: str
    current_timestamp_utc: str
    isolation_forest_raw_score: float
    normalized_risk_probability: float
    dominant_anomalous_features: list[str]
    model_confidence_interval: list[float]
    # Extra convenience fields for the frontend risk-tier colouring
    risk_tier: str = Field(
        ..., description="GREEN | YELLOW | ORANGE | RED based on agreed thresholds"
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int


class ModelHealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    feature_count: int
    server_utc: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _signals_from_request(req: Optional[LiveSignalsRequest]) -> LiveSignals:
    """Convert Pydantic LiveSignalsRequest → LiveSignals dataclass."""
    if req is None:
        return LiveSignals()

    obs_utc: Optional[datetime] = None
    if req.observation_utc:
        try:
            obs_utc = datetime.fromisoformat(req.observation_utc)
            if obs_utc.tzinfo is None:
                obs_utc = obs_utc.replace(tzinfo=timezone.utc)
        except ValueError:
            logger.warning("Could not parse observation_utc '%s' — using server time.", req.observation_utc)

    return LiveSignals(
        rainfall_mm_1h=req.rainfall_mm_1h,
        rainfall_mm_6h=req.rainfall_mm_6h,
        velocity_kmh=req.velocity_kmh,
        incident_count_2h=req.incident_count_2h,
        visibility_km=req.visibility_km,
        water_level_m=req.water_level_m,
        segment_load_factor=req.segment_load_factor,
        current_delay_min=req.current_delay_min,
        historical_avg_delay_min=req.historical_avg_delay_min,
        temp_celsius=req.temp_celsius,
        wind_speed_kmh=req.wind_speed_kmh,
        observation_utc=obs_utc,
    )


def _result_to_response(result: PredictionResult) -> PredictionResponse:
    d = result.to_dict()
    return PredictionResponse(
        **d,
        risk_tier=risk_tier(d["normalized_risk_probability"]),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Score a single route segment for disruption risk",
    description=(
        "Accepts a route segment object and optional live signals. "
        "Returns an IsolationForest anomaly score normalised to [0, 1] "
        "along with dominant anomalous features and a confidence interval."
    ),
)
async def predict(body: PredictRequest) -> PredictionResponse:
    """
    Main ML inference endpoint.  Called by the frontend (via Supabase edge
    function or direct API call) whenever a segment needs to be re-scored.
    """
    segment_dict = body.segment.model_dump()
    signals = _signals_from_request(body.signals)

    try:
        result = run_prediction(segment_dict, signals)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed for segment %s", body.segment.segment_id)
        raise HTTPException(status_code=500, detail="Internal model error") from exc

    return _result_to_response(result)


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Score multiple route segments in one call",
)
async def predict_batch(body: BatchPredictRequest) -> BatchPredictionResponse:
    """
    Batch scoring — useful for the Mapbox layer refresh that scores all NH
    segments at once.  Maximum 50 segments per request.
    """
    segments = [item.segment.model_dump() for item in body.items]
    signals_list = [_signals_from_request(item.signals) for item in body.items]

    try:
        results = run_batch_predictions(segments, signals_list)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail="Internal model error") from exc

    responses = [_result_to_response(r) for r in results]
    return BatchPredictionResponse(predictions=responses, total=len(responses))


@router.get(
    "/demo/chennai-replay",
    summary="Stream the Nov 2023 Chennai Floods simulation replay",
    description=(
        "Returns a newline-delimited JSON (NDJSON) stream of per-hour "
        "prediction results from T-12h to T+0 (road closure). "
        "The IsolationForest should flag disruption at T-4 to T-6h. "
        "Use ``delay_ms`` to control replay speed for live demos."
    ),
)
async def chennai_replay(
    delay_ms: int = Query(
        default=300,
        ge=0,
        le=5000,
        description="Milliseconds to wait between emitting each time-step (for live demo pacing)",
    ),
    full_report: bool = Query(
        default=False,
        description="If true, return a single JSON object with the full timeline instead of NDJSON stream",
    ),
) -> Any:
    """
    Streams the synthetic Nov 2023 Chennai Floods timeline through the
    prediction pipeline, emitting one NDJSON line per simulated hour.

    Frontend can consume this via EventSource / ReadableStream.
    """
    import json

    if full_report:
        # Synchronous full-report mode for demos / validation checks
        report = validate_chennai_scenario(verbose=False)
        return report

    # NH48 segment for the demo
    demo_segment = {
        "segment_id":                "seg-nh48-chennai-flood-demo",
        "nh_identifier":             "NH48",
        "start_node_latlon":         [13.0827, 80.2707],
        "end_node_latlon":           [12.9716, 79.9592],
        "base_distance_km":          62.0,
        "historical_delay_variance": 18.5,
    }

    timeline = generate_chennai_flood_timeline()

    async def _stream():
        model, scaler = load_model()
        for step in timeline:
            signals_kwargs = {k: v for k, v in step.items() if not k.startswith("_")}
            signals = LiveSignals(**signals_kwargs)

            try:
                result = run_prediction(demo_segment, signals)
                payload = result.to_dict()
                payload["risk_tier"] = risk_tier(payload["normalized_risk_probability"])
                payload["hours_before_closure"] = step["_hours_before_closure"]
                payload["event"] = step["_event"]
            except Exception as exc:
                payload = {"error": str(exc), "hours_before_closure": step["_hours_before_closure"]}

            yield json.dumps(payload) + "\n"

            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000.0)

    return StreamingResponse(
        _stream(),
        media_type="application/x-ndjson",
        headers={
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-cache",
        },
    )


@router.get(
    "/model/health",
    response_model=ModelHealthResponse,
    summary="Model health / readiness probe",
)
async def model_health() -> ModelHealthResponse:
    """
    Kubernetes / Cloud Run readiness probe.  Verifies model artefacts are
    loaded and returns basic metadata.
    """
    from data.feature_engineering import FEATURE_COLS

    try:
        model, _ = load_model()
        loaded = True
        model_type = type(model).__name__
        n_features = len(FEATURE_COLS)
    except Exception as exc:
        logger.error("Health check: model load failed — %s", exc)
        raise HTTPException(status_code=503, detail=f"Model not ready: {exc}") from exc

    return ModelHealthResponse(
        status="ok",
        model_loaded=loaded,
        model_type=model_type,
        feature_count=n_features,
        server_utc=datetime.now(timezone.utc).isoformat(),
    )
