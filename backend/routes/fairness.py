"""
FairChain Supplier Fairness Auditor API

Endpoints
---------
GET  /fairness/audit            – full audit metrics payload (Jaideep's modal contract)
GET  /fairness/audit/summary    – lightweight card-level summary
GET  /fairness/vendors          – paginated penalised vendor list
POST /fairness/audit/refresh    – re-run audit on latest supplier data (background)

JSON payload contract agreed with Jaideep:
    {
        "disparate_impact":      float,
        "parity_gap":            float,
        "audit_passed":          bool,
        "top_penalized_vendors": [...],
        "mitigation_result":     {...},
        "explainer":             {...},
    }
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import JSONResponse

# ── import auditor from its actual location (backend/ml/) ──────────────────
_ML_DIR = os.path.join(os.path.dirname(__file__), "..", "ml")
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

from fairness_auditor import FairnessAuditor  # noqa: E402

logger = logging.getLogger("fairchain.routes.fairness")

router = APIRouter(prefix="/fairness", tags=["Fairness Audit"])

# ── resolved path to the supplier CSV ─────────────────────────────────────
_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "supplier_database.csv"
)

# ---------------------------------------------------------------------------
# Singleton state — loaded once on first request, refreshable via POST
# ---------------------------------------------------------------------------
_auditor: Optional[FairnessAuditor] = None
_df: Optional[pd.DataFrame] = None
_cached_result: Optional[dict] = None


def _load_and_run() -> None:
    """(Re-)load CSV and compute all audit metrics into module-level cache."""
    global _auditor, _df, _cached_result

    logger.info("Loading supplier data from %s …", _DATA_PATH)
    df = pd.read_csv(_DATA_PATH)
    auditor = FairnessAuditor()

    metrics     = auditor.calculate_metrics(df)
    explainer   = auditor.generate_explainer(df)
    comparison  = auditor.compare_before_after(df)

    # Top penalised vendors — unprivileged group, sorted by penalty desc
    df_copy = df.copy()
    df_copy["penalty"] = df_copy["true_performance_score"] - df_copy["ai_trust_score"]
    top_vendors = (
        df_copy[df_copy["location_tier"] == 0]
        .nlargest(10, "penalty")
        [[
            "supplier_id", "location_tier", "business_size", "owner_gender",
            "years_active", "true_performance_score", "ai_trust_score", "penalty",
        ]]
        .round(2)
        .to_dict(orient="records")
    )

    _auditor = auditor
    _df = df
    _cached_result = {
        # ── Core fields Jaideep needs ──────────────────────────────────────
        "disparate_impact":      metrics["disparate_impact_ratio"],
        "parity_gap":            metrics["raw_score_gap"],
        "audit_passed":          not metrics["audit_failed_80_percent_rule"],
        # ── Extended metrics ───────────────────────────────────────────────
        "statistical_parity_difference": metrics["statistical_parity_difference"],
        "privileged_avg_score":          metrics["privileged_avg_score"],
        "unprivileged_avg_score":        metrics["unprivileged_avg_score"],
        "threshold_applied":             0.80,
        "total_suppliers_audited":       len(df),
        # ── Vendor list ────────────────────────────────────────────────────
        "top_penalized_vendors": top_vendors,
        # ── Mitigation before/after ────────────────────────────────────────
        "mitigation_result": {
            "disparate_impact_before":  comparison["before_mitigation"]["disparate_impact_ratio"],
            "disparate_impact_after":   comparison["after_mitigation"]["disparate_impact_ratio"],
            "improvement_delta":        comparison["improvements"]["disparate_impact_delta"],
            "score_gap_reduced_by_pts": comparison["improvements"]["score_gap_reduced_by_pts"],
            "score_lift_applied_pts":   comparison["improvements"]["score_lift_applied_pts"],
            "reweighing_boost_factor":  comparison["improvements"]["reweighing_boost_factor"],
            "audit_now_passes":         comparison["improvements"]["audit_now_passes"],
            "mitigation_algorithm":     comparison["improvements"]["mitigation_algorithm"],
        },
        # ── Full explainer for 'Why did audit fail?' modal ─────────────────
        "explainer": explainer,
    }
    logger.info(
        "Audit complete | DI=%.4f | parity_gap=%.2f | passed=%s",
        metrics["disparate_impact_ratio"],
        metrics["raw_score_gap"],
        not metrics["audit_failed_80_percent_rule"],
    )


def _get_result() -> dict:
    """Lazy-init: run audit on first call."""
    if _cached_result is None:
        _load_and_run()
    return _cached_result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/audit",
    summary="Full Fairness Audit Payload",
    description=(
        "Returns the complete fairness audit result. "
        "Payload includes: disparate_impact, parity_gap, audit_passed, "
        "top_penalized_vendors, mitigation_result, and the explainer object "
        "consumed by the Dashboard Fairness Modal."
    ),
)
async def get_full_audit() -> JSONResponse:
    """
    Primary endpoint consumed by Jaideep's Fairness Scorecard dashboard component.
    All field names match the agreed JSON shape.
    """
    try:
        result = _get_result()
        logger.info("Full audit payload served.")
        return JSONResponse(content=result)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=f"Supplier database not found at '{_DATA_PATH}'. Run dataset_generator.py first.",
        )
    except Exception as exc:
        logger.exception("Audit computation error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Audit failed: {exc}")


@router.get(
    "/audit/summary",
    summary="Lightweight Audit Summary (for dashboard cards)",
    description="Returns only top-level pass/fail status and key metrics. Cheap to poll.",
)
async def get_audit_summary() -> JSONResponse:
    """Minimal payload for the header summary cards — avoids sending the full vendor list."""
    result = _get_result()
    return JSONResponse(
        content={
            "disparate_impact":        result["disparate_impact"],
            "parity_gap":              result["parity_gap"],
            "audit_passed":            result["audit_passed"],
            "threshold_applied":       result["threshold_applied"],
            "total_suppliers_audited": result["total_suppliers_audited"],
            "privileged_avg_score":    result["privileged_avg_score"],
            "unprivileged_avg_score":  result["unprivileged_avg_score"],
        }
    )


@router.get(
    "/vendors",
    summary="Paginated Penalised Vendor List",
    description=(
        "Returns vendors flagged as unfairly penalised (true_performance > ai_trust_score). "
        "Supports pagination via skip and limit."
    ),
)
async def get_penalised_vendors(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
    location_tier: Optional[int] = Query(
        default=None,
        description="Filter by location_tier: 0 = Tier-3 flood-prone, 1 = Tier-1 metro",
    ),
) -> JSONResponse:
    """Paginated endpoint so the frontend table doesn't load all rows at once."""
    if _df is None:
        _get_result()  # ensure data is loaded

    df_copy = _df.copy()  # type: ignore[union-attr]
    df_copy["penalty"] = df_copy["true_performance_score"] - df_copy["ai_trust_score"]
    penalised = df_copy[df_copy["penalty"] > 0].copy()

    if location_tier is not None:
        penalised = penalised[penalised["location_tier"] == location_tier]

    penalised = penalised.sort_values("penalty", ascending=False)
    total = len(penalised)
    sliced = penalised.iloc[skip : skip + limit]

    return JSONResponse(
        content={
            "total":          total,
            "skip":           skip,
            "limit":          limit,
            "location_tier":  location_tier,
            "vendors": sliced[
                [
                    "supplier_id", "location_tier", "business_size", "owner_gender",
                    "years_active", "true_performance_score", "ai_trust_score", "penalty",
                ]
            ].round(2).to_dict(orient="records"),
        }
    )


@router.post(
    "/audit/refresh",
    summary="Re-run Audit on Latest Supplier Data",
    status_code=202,
)
async def refresh_audit(background_tasks: BackgroundTasks) -> JSONResponse:
    """Non-blocking refresh — audit reruns in background, current results remain available."""
    def _refresh():
        logger.info("Background audit refresh started.")
        _load_and_run()
        logger.info("Background audit refresh complete.")

    background_tasks.add_task(_refresh)
    return JSONResponse(
        status_code=202,
        content={"message": "Audit refresh queued. Results will update shortly."},
    )