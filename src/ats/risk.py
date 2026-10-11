from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any

@dataclass(frozen=True)
class RiskProfile:
    level: str = "medium"
    max_position_risk_pct: float = 0.50
    max_daily_loss_pct: float = 1.50
    min_conviction: float = 0.60
    max_model_disagreement: float = 0.15

PROFILES = {
    "low": RiskProfile("low", 0.25, 0.75, 0.70, 0.10),
    "medium": RiskProfile("medium", 0.50, 1.50, 0.60, 0.15),
    "high": RiskProfile("high", 1.00, 2.50, 0.55, 0.20),
}

class RiskEngine:
    def __init__(self, store):
        self.store = store
        saved = store.get_state("risk_profile", asdict(PROFILES["medium"]))
        self.profile = RiskProfile(**saved)

    def set_level(self, level: str) -> dict[str, Any]:
        level = level.lower()
        if level not in PROFILES:
            raise ValueError("Risk level must be low, medium, or high")
        self.profile = PROFILES[level]
        self.store.set_state("risk_profile", asdict(self.profile), __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat())
        return asdict(self.profile)

    def decision(self, forecast: dict[str, Any]) -> dict[str, Any]:
        p = float(forecast.get("conviction", 0.0))
        direction = float(forecast["forecast"]["p_up"])
        weights = forecast.get("weights", {})
        disagreement = float(forecast.get("model_disagreement", 0.0))
        strong = p >= self.profile.min_conviction
        if strong and direction >= 0.55 and disagreement <= self.profile.max_model_disagreement:
            bias = "LONG BIAS"
        elif strong and direction <= 0.45 and disagreement <= self.profile.max_model_disagreement:
            bias = "SHORT BIAS"
        else:
            bias = "WAIT"
        return {
            "bias": bias,
            "risk_level": self.profile.level,
            "max_position_risk_pct": self.profile.max_position_risk_pct,
            "max_daily_loss_pct": self.profile.max_daily_loss_pct,
            "conviction_gate": self.profile.min_conviction,
            "model_disagreement": disagreement,
            "reason": "Forecast/risk gates passed" if bias != "WAIT" else "Forecast confidence or model agreement below risk gate",
            "execution_enabled": False,
        }
