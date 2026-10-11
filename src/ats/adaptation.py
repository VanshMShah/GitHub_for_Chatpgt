from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class Diagnosis:
    code: str
    severity: str
    reason: str
    action: str


class ResidualLearner:
    """Tiny recursive least-squares residual corrector.

    WHY: only systematic residual structure should be learned, not isolated misses.
    """

    def __init__(self, n_features: int, forgetting: float = 0.985, state: dict[str, Any] | None = None):
        self.n = int(n_features)
        self.lam = float(forgetting)
        state = state or {}
        self.w = np.asarray(state.get("w", [0.0] * self.n), dtype=float)
        self.P = np.asarray(state.get("P", (np.eye(self.n) * 10.0).tolist()), dtype=float)
        self.samples = int(state.get("samples", 0))
        self.enabled = bool(state.get("enabled", False))
        if self.w.shape != (self.n,) or self.P.shape != (self.n, self.n):
            self.w = np.zeros(self.n)
            self.P = np.eye(self.n) * 10.0
            self.samples = 0
            self.enabled = False

    def predict(self, x: np.ndarray) -> float:
        return float(self.w @ np.asarray(x, dtype=float))

    def update(self, x: np.ndarray, residual: float) -> float:
        x = np.asarray(x, dtype=float)
        px = self.P @ x
        denom = self.lam + float(x @ px)
        gain = px / max(denom, 1e-12)
        pred = float(self.w @ x)
        self.w += gain * (float(residual) - pred)
        self.P = (self.P - np.outer(gain, px)) / self.lam
        self.P = 0.5 * (self.P + self.P.T)
        self.samples += 1
        return pred

    def state(self) -> dict[str, Any]:
        return {
            "w": self.w.tolist(),
            "P": self.P.tolist(),
            "lambda": self.lam,
            "samples": self.samples,
            "enabled": self.enabled,
        }


def diagnose(
    realized: float,
    forecast_mean: float,
    forecast_vol: float,
    q05: float,
    q95: float,
    p_up: float,
    recent_losses: list[float] | None = None,
    shock: bool = False,
) -> Diagnosis:
    recent_losses = recent_losses or []
    resid = float(realized - forecast_mean)
    scale = max(float(forecast_vol), 1e-6)

    if shock and abs(resid) > max(3.0 * scale, 0.015):
        return Diagnosis("EXOGENOUS_SHOCK", "info", "Large residual coincided with an external shock.", "Preserve base model; activate shock-aware policy.")
    if realized < q05 or realized > q95:
        return Diagnosis("DISTRIBUTION_MISS", "high", "Outcome fell outside the 90% prediction band.", "Reassess dispersion and tail calibration.")
    if abs(resid) > 2.0 * scale:
        return Diagnosis("MAGNITUDE_MISS", "medium", "Movement magnitude was badly underestimated.", "Evaluate residual and volatility correction.")
    direction_wrong = (realized > 0) != (p_up >= 0.5)
    if direction_wrong:
        return Diagnosis("DIRECTION_MISS", "medium", "Forecast direction disagreed with the realized return.", "Update directional expert skill; do not retrain on one miss.")
    if len(recent_losses) >= 12:
        prior = float(np.mean(recent_losses[:-5]))
        recent = float(np.mean(recent_losses[-5:]))
        if prior > 1e-9 and recent > 1.35 * prior:
            return Diagnosis("SYSTEMATIC_DRIFT", "high", "Recent loss is persistently above its prior rolling level.", "Increase adaptation pressure and evaluate challengers.")
    return Diagnosis("NORMAL", "none", "No persistent forecast failure detected.", "No structural change.")
