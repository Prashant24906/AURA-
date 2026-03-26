"""
parking_predictor.py
--------------------
Main entry point for the Smart Parking Prediction module.
Combines:
  1. Base zone availability prediction
  2. Event surge adjustment
  3. Fine risk alert
  4. Vehicle type filtering
  5. Safety scoring

Returns a ranked list of recommended parking zones for the user.
Integrates with the team's FastAPI backend via predict_parking().
"""

import json
from datetime import datetime
from pathlib import Path

from event_surge_predictor import EventSurgePredictor
from fine_risk_alert import FineRiskAnalyser


# ─── Base occupancy model (simple rule-based simulation) ─────────────────────
# In your full project, replace this with your trained ML model
# (e.g. RandomForestRegressor trained on historical parking CSV data)

BASE_OCCUPANCY = {
    # zone_id: {day_type: {time_slot: occupancy_pct}}
    "Z001": {"weekday": {"morning": 35, "afternoon": 55, "evening": 65},
             "weekend": {"morning": 55, "afternoon": 75, "evening": 85}},
    "Z002": {"weekday": {"morning": 40, "afternoon": 60, "evening": 70},
             "weekend": {"morning": 60, "afternoon": 80, "evening": 88}},
    "Z003": {"weekday": {"morning": 50, "afternoon": 65, "evening": 75},
             "weekend": {"morning": 65, "afternoon": 85, "evening": 92}},
    "Z004": {"weekday": {"morning": 20, "afternoon": 35, "evening": 50},
             "weekend": {"morning": 30, "afternoon": 50, "evening": 65}},
    "Z005": {"weekday": {"morning": 90, "afternoon": 95, "evening": 98},
             "weekend": {"morning": 95, "afternoon": 99, "evening": 100}},
    "Z006": {"weekday": {"morning": 80, "afternoon": 90, "evening": 95},
             "weekend": {"morning": 90, "afternoon": 97, "evening": 99}},
}

def get_time_slot(hour: int) -> str:
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    else:
        return "evening"

def get_day_type(dt: datetime) -> str:
    return "weekend" if dt.weekday() >= 5 else "weekday"


# ─── Main predictor class ─────────────────────────────────────────────────────

class SmartParkingPredictor:
    """
    Combines surge prediction + fine risk + zone availability
    to recommend the best parking spots for a user.
    """

    def __init__(
        self,
        zones_path: str = "data/zones.json",
        events_path: str = "data/events.json",
    ):
        with open(Path(zones_path)) as f:
            self.zones = json.load(f)

        self.surge_predictor = EventSurgePredictor(events_path=events_path)
        self.risk_analyser = FineRiskAnalyser(zones_path=zones_path)

    def predict_parking(
        self,
        dest_lat: float,
        dest_lng: float,
        vehicle_type: str = "4w",           # "2w" or "4w"
        query_dt: datetime | None = None,
        max_walk_min: int = 15,
        include_risky: bool = False,         # set True to show risky zones too
    ) -> dict:
        """
        Full prediction pipeline. Returns ranked zone recommendations.

        Parameters
        ----------
        dest_lat, dest_lng  : Where the user wants to go
        vehicle_type        : "2w" or "4w"
        query_dt            : Visit date and time
        max_walk_min        : Maximum acceptable walk time (minutes)
        include_risky       : Include high-risk zones in output (with warnings)

        Returns
        -------
        dict with:
            surge_report    – event surge details
            recommendations – ranked list of zones with full details
            destination_occupancy – predicted % full at main destination
            summary         – one-line human summary
        """
        if query_dt is None:
            query_dt = datetime.now()

        # ── Step 1: Get event surge report ────────────────────────────────
        surge = self.surge_predictor.get_surge(dest_lat, dest_lng, query_dt)
        surge_mult = surge["surge_multiplier"]

        # ── Step 2: Get time context ──────────────────────────────────────
        day_type = get_day_type(query_dt)
        time_slot = get_time_slot(query_dt.hour)

        # ── Step 3: Get fine risk for all zones ───────────────────────────
        risk_alerts = {
            a.zone_id: a
            for a in self.risk_analyser.analyse_all_zones(vehicle_type)
        }

        # ── Step 4: Score and rank each zone ─────────────────────────────
        ranked = []
        for zone in self.zones:
            zid = zone["zone_id"]
            alert = risk_alerts.get(zid)
            if not alert:
                continue

            # Filter by walk distance
            walk = zone.get("walk_to_market_min", 99)
            if walk > max_walk_min:
                continue

            # Filter out illegal zones (unless user asked to include)
            if not include_risky and not alert.is_legal:
                continue

            # Filter by vehicle type
            v_notes = alert.vehicle_specific.get(vehicle_type, {})
            if not v_notes.get("allowed", False):
                continue

            # Get base occupancy and apply surge
            base_occ = BASE_OCCUPANCY.get(zid, {}).get(day_type, {}).get(time_slot, 50)
            adjusted_occ = min(base_occ * surge_mult, 100)
            availability = round(100 - adjusted_occ, 1)

            # Composite recommendation score (higher = better)
            rec_score = self._rec_score(
                availability=availability,
                risk_score=alert.risk_score,
                safety=zone.get("safety_score", 50),
                walk=walk,
                cctv=zone.get("cctv", False),
            )

            ranked.append({
                "zone_id": zid,
                "zone_name": zone["name"],
                "rec_score": rec_score,
                "availability_pct": availability,
                "base_occupancy_pct": base_occ,
                "surge_adjusted": surge_mult > 1.0,
                "walk_to_destination_min": walk,
                "risk_level": alert.risk_level,
                "risk_score": alert.risk_score,
                "risk_label": alert.risk_label,
                "estimated_fine": alert.estimated_fine,
                "tow_risk": alert.tow_risk,
                "safety_score": zone.get("safety_score", 50),
                "cctv": zone.get("cctv", False),
                "lighting": zone.get("lighting", "unknown"),
                "fee_per_hour": zone.get("fee_per_hour", 0),
                "accessible": zone.get("accessible", False),
                "time_restriction": zone.get("time_restrictions"),
                "warnings": alert.warnings,
                "advice": alert.advice,
                "vehicle_capacity": v_notes.get("capacity", 0),
                "color": self._avail_color(availability, alert.risk_level),
            })

        # Sort by composite recommendation score (descending)
        ranked.sort(key=lambda x: x["rec_score"], reverse=True)

        # ── Step 5: Build human-readable summary ─────────────────────────
        summary = self._build_summary(ranked, surge, vehicle_type, day_type)

        return {
            "query_datetime": query_dt.strftime("%Y-%m-%d %H:%M"),
            "vehicle_type": "Two-wheeler" if vehicle_type == "2w" else "Four-wheeler",
            "surge_report": surge,
            "recommendations": ranked,
            "total_viable_zones": len(ranked),
            "summary": summary,
        }

    # ── Scoring helpers ───────────────────────────────────────────────────

    def _rec_score(
        self,
        availability: float,
        risk_score: int,
        safety: int,
        walk: int,
        cctv: bool,
    ) -> float:
        """
        Composite score for ranking zones.
        Higher is better.
        Weights: availability (40%), safety (25%), risk (20%), walk (15%)
        """
        avail_score  = availability * 0.40
        safety_score = safety * 0.25
        risk_score_w = (100 - risk_score) * 0.20
        walk_score   = max(0, (15 - walk)) / 15 * 100 * 0.15
        cctv_bonus   = 3 if cctv else 0
        return round(avail_score + safety_score + risk_score_w + walk_score + cctv_bonus, 2)

    def _avail_color(self, avail: float, risk_level: str) -> str:
        if risk_level in ("high", "critical"):
            return "red"
        if avail >= 30:
            return "green"
        if avail >= 10:
            return "yellow"
        return "red"

    def _build_summary(
        self, ranked: list[dict], surge: dict, vehicle_type: str, day_type: str
    ) -> str:
        vt = "two-wheeler" if vehicle_type == "2w" else "four-wheeler"
        if not ranked:
            return (
                f"No suitable {vt} parking found within walking distance. "
                "Consider parking farther away or using public transport."
            )
        best = ranked[0]
        surge_note = ""
        if surge["surge_level"] in ("high", "critical"):
            surge_note = (
                f" Note: {surge['surge_label']} due to nearby event — "
                f"arrive {surge['adjusted_arrival']}."
            )
        return (
            f"Best option for your {vt}: {best['zone_name']} — "
            f"{best['availability_pct']}% available, "
            f"{best['walk_to_destination_min']} min walk, "
            f"risk: {best['risk_level']}.{surge_note}"
        )


# ─── Quick demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = SmartParkingPredictor()

    print("===== SMART PARKING SYSTEM =====\n")

    # 📍 Location selection (Pune areas)
    locations = {
        "1": {"name": "MG Road", "lat": 18.5210, "lng": 73.8570},
        "2": {"name": "FC Road", "lat": 18.5204, "lng": 73.8410},
        "3": {"name": "Deccan Gymkhana", "lat": 18.5196, "lng": 73.8412},
        "4": {"name": "Shivajinagar", "lat": 18.5308, "lng": 73.8475},
    }

    print("Select destination:")
    for key, loc in locations.items():
        print(f"{key}. {loc['name']}")

    loc_choice = input("Enter choice (1-4): ")
    selected_loc = locations.get(loc_choice)

    if not selected_loc:
        print("❌ Invalid location choice")
        exit()

    dest_lat = selected_loc["lat"]
    dest_lng = selected_loc["lng"]

    # 🚗 Vehicle type input
    print("\nSelect vehicle type:")
    print("1. Two-wheeler")
    print("2. Four-wheeler")

    vehicle_choice = input("Enter choice (1/2): ")

    if vehicle_choice == "1":
        vehicle_type = "2w"
    elif vehicle_choice == "2":
        vehicle_type = "4w"
    else:
        print("❌ Invalid vehicle choice")
        exit()

    # 🕒 Date & time input
    user_time = input("\nEnter date & time (YYYY-MM-DD HH:MM): ")

    try:
        query_dt = datetime.strptime(user_time, "%Y-%m-%d %H:%M")
    except ValueError:
        print("❌ Invalid date/time format")
        exit()

    print(f"\n🔄 Checking parking near {selected_loc['name']}...\n")

    # 🔮 Run prediction
    result = predictor.predict_parking(
        dest_lat=dest_lat,
        dest_lng=dest_lng,
        vehicle_type=vehicle_type,
        query_dt=query_dt,
    )

    # 📊 Output
    print("\n===== SMART PARKING PREDICTION RESULT =====\n")
    print(f"Location     : {selected_loc['name']}")
    print(f"Time         : {result['query_datetime']}")
    print(f"Vehicle      : {result['vehicle_type']}")
    print(f"Surge level  : {result['surge_report']['surge_level'].upper()}")
    print(f"Summary      : {result['summary']}")

    if result["total_viable_zones"] == 0:
        print("\n⚠️ No parking spaces available nearby.\n")
        exit()

    print(f"\nTop {min(3, len(result['recommendations']))} Recommended Zones:\n")

    for i, zone in enumerate(result["recommendations"][:3], 1):
        print(f"  #{i} {zone['zone_name']}")
        print(f"      Availability : {zone['availability_pct']}%")
        print(f"      Walk time    : {zone['walk_to_destination_min']} min")
        print(f"      Safety score : {zone['safety_score']}/100")
        print(f"      Risk level   : {zone['risk_level']} (score: {zone['risk_score']}/100)")
        print(f"      Fine risk    : ₹{zone['estimated_fine']} if parked illegally")
        print(f"      Tow risk     : {'YES' if zone['tow_risk'] else 'No'}")
        print(f"      Fee          : ₹{zone['fee_per_hour']}/hr" if zone["fee_per_hour"] else "      Fee          : Free")

        if zone["warnings"]:
            for w in zone["warnings"]:
                print(f"      ! {w}")

        print(f"      Advice       : {zone['advice']}")
        print()
