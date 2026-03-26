"""
event_surge_predictor.py
------------------------
Detects nearby events for a given location and date/time,
then predicts how much extra parking demand (surge) will occur.
Adjusts zone availability predictions accordingly.
"""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path


# ─── Helpers ────────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate straight-line distance in km between two GPS coordinates."""
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def is_peak_hour(current_hour: int, peak_hours: list[str], window_hrs: int = 2) -> bool:
    """Return True if current hour is within `window_hrs` of any event peak hour."""
    for ph in peak_hours:
        peak_h = int(ph.split(":")[0])
        if abs(current_hour - peak_h) <= window_hrs:
            return True
    return False


# ─── Core class ─────────────────────────────────────────────────────────────

class EventSurgePredictor:
    """
    Loads event data and predicts surge multiplier for a given
    destination location and query datetime.
    """

    EVENT_TYPE_ICONS = {
        "festival": "Festival",
        "sports": "Sports match",
        "market": "Special market",
        "food_festival": "Food festival",
        "concert": "Concert",
    }

    SURGE_LEVELS = {
        (1.0, 1.5): ("normal",   "Normal demand",             "green"),
        (1.5, 2.0): ("moderate", "Moderately busy",           "yellow"),
        (2.0, 3.0): ("high",     "High demand – plan early",  "orange"),
        (3.0, 999): ("critical", "Extreme surge – very full", "red"),
    }

    def __init__(self, events_path: str = "data/events.json"):
        path = Path(events_path)
        if not path.exists():
            raise FileNotFoundError(f"Events file not found: {events_path}")
        with open(path) as f:
            self.events: list[dict] = json.load(f)

    # ── Public API ────────────────────────────────────────────────────────

    def get_surge(
        self,
        dest_lat: float,
        dest_lng: float,
        query_dt: datetime | None = None,
        search_radius_km: float = 2.0,
    ) -> dict:
        """
        Main method. Returns a surge report for the destination.

        Parameters
        ----------
        dest_lat, dest_lng  : GPS coords of where the user wants to park
        query_dt            : The date/time the user plans to visit
                              (defaults to now)
        search_radius_km    : How far around the destination to look for events

        Returns
        -------
        dict with keys:
            surge_multiplier  – float ≥ 1.0  (multiply base occupancy by this)
            surge_level       – "normal" / "moderate" / "high" / "critical"
            surge_label       – human-readable description
            color             – UI color hint
            active_events     – list of nearby events affecting demand
            warnings          – list of alert strings to show the user
            recommendations   – list of actionable advice strings
            adjusted_arrival  – suggested arrival datetime (string)
        """
        if query_dt is None:
            query_dt = datetime.now()

        nearby = self._find_nearby_events(dest_lat, dest_lng, query_dt, search_radius_km)

        # Combine multipliers: events stack but with diminishing returns
        combined_multiplier = 1.0
        for ev in nearby:
            m = ev["effective_multiplier"]
            combined_multiplier = combined_multiplier + (m - 1.0) * 0.8  # diminishing stack

        combined_multiplier = round(min(combined_multiplier, 5.0), 2)  # cap at 5×

        level, label, color = self._classify_surge(combined_multiplier)

        warnings = self._build_warnings(nearby, combined_multiplier)
        recommendations = self._build_recommendations(nearby, combined_multiplier, query_dt)
        adjusted_arrival = self._suggest_arrival(nearby, query_dt, combined_multiplier)

        return {
            "surge_multiplier": combined_multiplier,
            "surge_level": level,
            "surge_label": label,
            "color": color,
            "active_events": nearby,
            "warnings": warnings,
            "recommendations": recommendations,
            "adjusted_arrival": adjusted_arrival,
            "query_datetime": query_dt.strftime("%Y-%m-%d %H:%M"),
            "events_found": len(nearby),
        }

    def adjust_zone_availability(self, zone_avail_pct: float, surge_multiplier: float) -> float:
        """
        Given a predicted base availability % and surge multiplier,
        return the adjusted (lower) availability %.

        Example: 60% available at 2× surge → 30% adjusted
        """
        adjusted = zone_avail_pct / surge_multiplier
        return round(max(adjusted, 0.0), 1)

    # ── Private helpers ───────────────────────────────────────────────────

    def _find_nearby_events(
        self, lat: float, lng: float, dt: datetime, radius_km: float
    ) -> list[dict]:
        """Find all events within radius_km that are active on the given date."""
        date_str = dt.strftime("%Y-%m-%d")
        results = []

        for ev in self.events:
            # Check date match
            if date_str not in ev.get("dates", []):
                continue

            # Check distance
            dist = haversine_km(lat, lng, ev["lat"], ev["lng"])
            effective_radius = ev.get("radius_km", 1.0)
            if dist > max(radius_km, effective_radius):
                continue

            # Calculate effective multiplier (lower if far away)
            distance_factor = max(0.3, 1.0 - (dist / effective_radius) * 0.5)
            at_peak = is_peak_hour(dt.hour, ev.get("peak_hours", []))
            peak_factor = 1.0 if at_peak else 0.6

            effective_mult = 1.0 + (ev["demand_multiplier"] - 1.0) * distance_factor * peak_factor

            results.append({
                "event_id": ev["event_id"],
                "name": ev["name"],
                "type": ev["type"],
                "type_label": self.EVENT_TYPE_ICONS.get(ev["type"], ev["type"].title()),
                "location": ev["location"],
                "distance_km": round(dist, 2),
                "peak_hours": ev.get("peak_hours", []),
                "at_peak_hour": at_peak,
                "base_multiplier": ev["demand_multiplier"],
                "effective_multiplier": round(effective_mult, 2),
                "expected_crowd": ev.get("expected_crowd", "N/A"),
            })

        # Sort by impact (highest multiplier first)
        results.sort(key=lambda x: x["effective_multiplier"], reverse=True)
        return results

    def _classify_surge(self, multiplier: float) -> tuple[str, str, str]:
        for (low, high), (level, label, color) in self.SURGE_LEVELS.items():
            if low <= multiplier < high:
                return level, label, color
        return "critical", "Extreme surge", "red"

    def _build_warnings(self, events: list[dict], multiplier: float) -> list[str]:
        warnings = []
        for ev in events:
            icon = ev["type_label"]
            warnings.append(
                f"{icon} nearby: '{ev['name']}' at {ev['location']} "
                f"({ev['distance_km']} km away) — "
                f"{ev['effective_multiplier']}× normal demand"
            )
            if ev["at_peak_hour"]:
                warnings.append(
                    f"You are visiting during peak hours of '{ev['name']}'. "
                    f"Expect maximum crowds around {', '.join(ev['peak_hours'])}."
                )
        if multiplier >= 3.0:
            warnings.append(
                "CRITICAL: Parking demand is extremely high. "
                "Legal spots near destination may be completely full."
            )
        return warnings

    def _build_recommendations(
        self, events: list[dict], multiplier: float, dt: datetime
    ) -> list[str]:
        recs = []
        if multiplier >= 2.0:
            recs.append("Arrive at least 45–60 minutes earlier than usual.")
            recs.append("Consider parking farther away (Zone D) and walking.")
        if multiplier >= 3.0:
            recs.append("Use public transport or auto-rickshaw if possible.")
            recs.append("Pre-book a paid parking slot if available in the area.")
        for ev in events:
            if ev["type"] == "festival":
                recs.append(
                    f"Festival shuttle services may be available near {ev['location']}. "
                    f"Check local transport apps."
                )
            if ev["type"] == "sports":
                recs.append(
                    "Stadium parking fills up 2 hours before match time. "
                    "Look for residential lane parking 1–2 km away."
                )
        if not recs:
            recs.append("Normal conditions — parking should be easy to find.")
        return recs

    def _suggest_arrival(
        self, events: list[dict], dt: datetime, multiplier: float
    ) -> str:
        """Suggest an earlier arrival time to beat the surge."""
        if multiplier < 1.5:
            return dt.strftime("%H:%M (no change needed)")
        extra_minutes = int((multiplier - 1.0) * 25)  # scale: 2× → 25min earlier
        suggested = dt - timedelta(minutes=extra_minutes)
        return suggested.strftime(f"%H:%M (arrive {extra_minutes} min earlier)")


# ─── Quick demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = EventSurgePredictor(events_path="data/events.json")

    # Simulate: user wants to go to MG Road market on Ganesh festival day
    query = datetime(2024, 9, 8, 14, 0)  # 2pm on festival day
    result = predictor.get_surge(
        dest_lat=18.5210,
        dest_lng=73.8570,
        query_dt=query,
    )

    print("\n===== EVENT SURGE PREDICTION =====")
    print(f"Date/Time      : {result['query_datetime']}")
    print(f"Events found   : {result['events_found']}")
    print(f"Surge level    : {result['surge_level'].upper()} ({result['surge_label']})")
    print(f"Surge multiplier: {result['surge_multiplier']}×")
    print(f"Suggested arrival: {result['adjusted_arrival']}")

    print("\n--- Warnings ---")
    for w in result["warnings"]:
        print(f"  ! {w}")

    print("\n--- Recommendations ---")
    for r in result["recommendations"]:
        print(f"  > {r}")

    print("\n--- Zone Availability After Surge Adjustment ---")
    base_availabilities = {"Zone A": 60, "Zone B": 45, "Zone D": 80}
    for zone, base in base_availabilities.items():
        adjusted = predictor.adjust_zone_availability(base, result["surge_multiplier"])
        print(f"  {zone}: {base}% → {adjusted}% (after {result['surge_multiplier']}× surge)")
