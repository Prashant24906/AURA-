"""
fine_risk_alert.py
------------------
Analyzes legality, risk, and penalties for parking zones.
"""

class FineRiskAlert:
    def __init__(
        self,
        zone_id,
        is_legal,
        risk_score,
        risk_level,
        risk_label,
        tow_risk,
        vehicle_specific,
        warnings,
        advice,
    ):
        self.zone_id = zone_id
        self.is_legal = is_legal
        self.risk_score = risk_score
        self.risk_level = risk_level
        self.risk_label = risk_label
        self.tow_risk = tow_risk
        self.vehicle_specific = vehicle_specific
        self.warnings = warnings
        self.advice = advice

        # ✅ FINAL RULE: Fixed fine
        self.estimated_fine = 500 if not is_legal else 0


class FineRiskAnalyser:
    def __init__(self, zones_path="data/zones.json"):
        import json
        from pathlib import Path

        with open(Path(zones_path)) as f:
            self.zones = json.load(f)

    def analyse_all_zones(self, vehicle_type):
        alerts = []

        for zone in self.zones:
            zid = zone["zone_id"]

            # Example rules (keep your existing logic if different)
            is_legal = not zone.get("no_parking", False)

            risk_score = 20 if is_legal else 80
            risk_level = "safe" if is_legal else "high"
            risk_label = "Low risk" if is_legal else "High risk"

            tow_risk = not is_legal and zone.get("tow_zone", False)

            vehicle_specific = {
                "2w": {"allowed": True, "capacity": 20},
                "4w": {"allowed": True, "capacity": 10},
            }

            warnings = []
            if not is_legal:
                warnings.append("Illegal parking zone")

            advice = (
                "This zone is safe and legal. Go ahead and park here."
                if is_legal
                else "Avoid parking here — fine applicable."
            )

            alert = FineRiskAlert(
                zone_id=zid,
                is_legal=is_legal,
                risk_score=risk_score,
                risk_level=risk_level,
                risk_label=risk_label,
                tow_risk=tow_risk,
                vehicle_specific=vehicle_specific,
                warnings=warnings,
                advice=advice,
            )

            alerts.append(alert)

        return alerts