from __future__ import annotations

import math
from typing import Any
from src.ats.models import Forecast


DEFAULT_WEIGHTS = {
    "persistence": 0.20,
    "logistic": 0.20,
    "lightgbm_compatible": 0.30,
    "arima_gjr_garch": 0.30,
}


class DMAPolicy:
    """Bounded dynamic model averaging with explicit prior/learned state."""

    def __init__(self, state: dict[str, Any] | None = None):
        state = state or {}
        self.version = int(state.get("version", 1))
        self.decay = float(state.get("decay", 0.97))
        self.update_rate = float(state.get("update_rate", 0.20))
        saved_priors = state.get("prior_weights", {})
        self.prior_weights = {k: float(saved_priors.get(k, v)) for k, v in DEFAULT_WEIGHTS.items()}
        self.weights = {k: float(state.get("weights", {}).get(k, v)) for k, v in self.prior_weights.items()}
        self.loss_ema = {k: float(v) for k, v in state.get("loss_ema", {}).items()}
        self.loss_ema.update({k: self.loss_ema.get(k, 1.0) for k in self.weights})
        self.sample_count = {k: int(v) for k, v in state.get("sample_count", {}).items()}
        self.sample_count.update({k: self.sample_count.get(k, 0) for k in self.weights})
        self.last_reason = str(state.get("last_reason", "initial policy"))

    def register_model(self, name: str, prior: float = 0.10) -> None:
        if name in self.weights:
            return
        prior = float(max(0.0, min(prior, 0.50)))
        old_total = sum(self.weights.values()) or 1.0
        scale = max(0.0, 1.0 - prior) / old_total
        self.weights = {k: v * scale for k, v in self.weights.items()}
        self.weights[name] = prior
        self.prior_weights[name] = prior
        self.loss_ema[name] = 1.0
        self.sample_count[name] = 0
        self.last_reason = f"registered {name} challenger as eligible expert"

    def context(self, latest: dict[str, float]) -> dict[str, float]:
        return latest

    @property
    def learned(self) -> bool:
        return any(v > 0 for v in self.sample_count.values())

    def choose(self, forecasts: dict[str, Forecast], context: dict[str, float]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for k in forecasts:
            skill = math.exp(-3.0 * max(self.loss_ema.get(k, 1.0), 0.0))
            prior = max(self.weights.get(k, self.prior_weights.get(k, 0.01)), 0.01)
            scores[k] = skill * prior
        total = sum(scores.values()) or 1.0
        return {k: v / total for k, v in scores.items()}

    def observe(self, realized: float, forecasts: dict[str, Forecast], reason: str = "settled forecast") -> dict[str, float]:
        for k, f in forecasts.items():
            if k not in self.weights:
                self.register_model(k, prior=0.05)
            scale = max(abs(f.volatility), 0.003)
            standardized_sq = min(((realized - f.mean) / scale) ** 2, 25.0) / 25.0
            p = min(max(f.p_up, 1e-4), 1.0 - 1e-4)
            up = 1 if realized > 0 else 0
            logloss = -(up * math.log(p) + (1 - up) * math.log(1 - p))
            loss = 0.65 * standardized_sq + 0.35 * min(logloss, 10.0) / 10.0
            prev = self.loss_ema.get(k, loss)
            self.loss_ema[k] = self.decay * prev + (1.0 - self.decay) * loss
            self.sample_count[k] = self.sample_count.get(k, 0) + 1

        raw = {k: math.exp(-3.0 * self.loss_ema.get(k, 1.0)) for k in self.weights if k in forecasts}
        total = sum(raw.values()) or 1.0
        learned = {k: v / total for k, v in raw.items()}
        for k in self.weights:
            target = learned.get(k, self.weights[k])
            self.weights[k] = (1.0 - self.update_rate) * self.weights[k] + self.update_rate * target
        total_w = sum(self.weights.values()) or 1.0
        self.weights = {k: v / total_w for k, v in self.weights.items()}
        self.version += 1
        self.last_reason = reason
        return self.weights

    def state(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "decay": self.decay,
            "update_rate": self.update_rate,
            "weights": self.weights,
            "prior_weights": self.prior_weights,
            "loss_ema": self.loss_ema,
            "sample_count": self.sample_count,
            "learned": self.learned,
            "weight_source": "learned" if self.learned else "prior",
            "last_reason": self.last_reason,
        }
