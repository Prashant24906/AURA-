"""
api.py
------
FastAPI routes for the Smart Parking module.
Plug this into your team's main FastAPI app with:

    from parking_module.api import router
    app.include_router(router, prefix="/parking")

Endpoints:
    POST /parking/predict        — main prediction endpoint
    GET  /parking/surge          — event surge check for a location
    GET  /parking/zone/{zone_id} — fine risk for a specific zone
    GET  /parking/zones/safe     — list all safe zones
"""

from datetime import datetime
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

from parking_predictor import SmartParkingPredictor
from event_surge_predictor import EventSurgePredictor
from fine_risk_alert import FineRiskAnalyser

router = APIRouter(tags=["Parking Prediction"])

# Instantiate modules once (shared across requests)
_predictor = SmartParkingPredictor(
    zones_path="data/zones.json",
    events_path="data/events.json",
)
_surge = EventSurgePredictor(events_path="data/events.json")
_risk  = FineRiskAnalyser(zones_path="data/zones.json")


# ─── Request / Response models ───────────────────────────────────────────────

class PredictRequest(BaseModel):
    dest_lat: float
    dest_lng: float
    vehicle_type: str = "4w"          # "2w" or "4w"
    visit_datetime: str | None = None  # "YYYY-MM-DD HH:MM" — defaults to now
    max_walk_min: int = 15


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.post("/predict")
def predict_parking(req: PredictRequest):
    """
    Main endpoint: returns ranked parking zones with surge + fine risk info.

    Example body:
        {
            "dest_lat": 18.5210,
            "dest_lng": 73.8570,
            "vehicle_type": "4w",
            "visit_datetime": "2024-09-08 14:00",
            "max_walk_min": 12
        }
    """
    try:
        dt = (
            datetime.strptime(req.visit_datetime, "%Y-%m-%d %H:%M")
            if req.visit_datetime
            else datetime.now()
        )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="visit_datetime must be in format YYYY-MM-DD HH:MM"
        )

    if req.vehicle_type not in ("2w", "4w"):
        raise HTTPException(status_code=400, detail="vehicle_type must be '2w' or '4w'")

    result = _predictor.predict_parking(
        dest_lat=req.dest_lat,
        dest_lng=req.dest_lng,
        vehicle_type=req.vehicle_type,
        query_dt=dt,
        max_walk_min=req.max_walk_min,
    )
    return result


@router.get("/surge")
def get_surge(
    lat: float = Query(..., description="Destination latitude"),
    lng: float = Query(..., description="Destination longitude"),
    visit_datetime: str = Query(
        None, description="YYYY-MM-DD HH:MM — defaults to now"
    ),
):
    """
    Check if any events are causing a parking surge at a location.

    Example: GET /parking/surge?lat=18.521&lng=73.857&visit_datetime=2024-09-08 14:00
    """
    try:
        dt = (
            datetime.strptime(visit_datetime, "%Y-%m-%d %H:%M")
            if visit_datetime
            else datetime.now()
        )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="visit_datetime must be in format YYYY-MM-DD HH:MM"
        )

    return _surge.get_surge(dest_lat=lat, dest_lng=lng, query_dt=dt)


@router.get("/zone/{zone_id}/risk")
def get_zone_risk(
    zone_id: str,
    vehicle_type: str = Query("4w", description="'2w' or '4w'"),
):
    """
    Get detailed fine risk alert for a specific zone.

    Example: GET /parking/zone/Z003/risk?vehicle_type=4w
    """
    try:
        alert = _risk.analyse_zone(zone_id=zone_id, vehicle_type=vehicle_type)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "zone_id": alert.zone_id,
        "zone_name": alert.zone_name,
        "risk_score": alert.risk_score,
        "risk_level": alert.risk_level,
        "risk_label": alert.risk_label,
        "color": alert.color,
        "is_legal": alert.is_legal,
        "tow_risk": alert.tow_risk,
        "estimated_fine_inr": alert.estimated_fine,
        "enforcement_active": alert.enforcement_active,
        "challan_last_30_days": alert.challan_last_30d,
        "time_restriction": alert.time_restriction,
        "warnings": alert.warnings,
        "advice": alert.advice,
        "vehicle_notes": alert.vehicle_specific,
    }


@router.get("/zones/safe")
def list_safe_zones(
    vehicle_type: str = Query("4w", description="'2w' or '4w'"),
    max_risk: int = Query(45, description="Max acceptable risk score (0–100)"),
):
    """
    List all zones that are safe and legal to park in.

    Example: GET /parking/zones/safe?vehicle_type=4w&max_risk=30
    """
    safe_zones = _risk.filter_safe_zones(vehicle_type=vehicle_type, max_risk_score=max_risk)
    return {
        "vehicle_type": vehicle_type,
        "max_risk_score_filter": max_risk,
        "count": len(safe_zones),
        "zones": [
            {
                "zone_id": z.zone_id,
                "zone_name": z.zone_name,
                "risk_score": z.risk_score,
                "risk_level": z.risk_level,
                "advice": z.advice,
            }
            for z in safe_zones
        ],
    }
