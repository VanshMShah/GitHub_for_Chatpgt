from __future__ import annotations

from dataclasses import dataclass
from math import floor

from src.ats.models import Forecast

from .risk import PaperRiskGuard, RiskDecision
from .types import OrderIntent, OrderType, QuoteEvent, Side


@dataclass(frozen=True)
class SignalConfig:
    expected_horizon: str = "1D"
    buy_threshold: float = 0.55
    sell_threshold: float = 0.45
    min_abs_return: float = 0.0
    target_notional: float = 1_000.0
    lot_size: int = 1
    allow_short: bool = False


@dataclass(frozen=True)
class SignalEvent:
    signal_id: str
    symbol: str
    timestamp_ns: int
    direction: int
    strength: float
    forecast_mean: float
    p_up: float
    horizon: str


class ForecastOrderBridge:
    """Converts a horizon-validated Forecast into a risk-checked market order."""

    def __init__(self, config: SignalConfig | None = None, risk: PaperRiskGuard | None = None):
        self.config = config or SignalConfig()
        self.risk = risk or PaperRiskGuard()

    def signal(self, forecast: Forecast, symbol: str, timestamp_ns: int, signal_id: str) -> SignalEvent:
        horizon = str(forecast.metadata.get("horizon", "unknown"))
        if horizon != self.config.expected_horizon:
            raise ValueError(f"forecast horizon {horizon!r} != expected {self.config.expected_horizon!r}")
        if forecast.metadata.get("instrument_key") not in (None, symbol):
            raise ValueError("forecast instrument mismatch")
        p = float(forecast.p_up)
        if p >= self.config.buy_threshold and forecast.mean >= self.config.min_abs_return:
            direction = 1
        elif p <= self.config.sell_threshold and forecast.mean <= -self.config.min_abs_return:
            direction = -1
        else:
            direction = 0
        strength = min(1.0, 2.0 * abs(p - 0.5))
        return SignalEvent(signal_id, symbol, timestamp_ns, direction, strength, float(forecast.mean), p, horizon)

    def order(self, signal: SignalEvent, quote: QuoteEvent, order_id: str, now_ns: int,
              current_notional: float = 0.0, realized_pnl: float = 0.0,
              current_quantity: int | None = None, other_notional: float = 0.0) -> tuple[OrderIntent | None, RiskDecision]:
        if signal.direction == 0:
            return None, RiskDecision(False, "no_trade_signal")
        reference = quote.ask if signal.direction > 0 else quote.bid
        if self.config.lot_size < 1:
            raise ValueError("lot_size must be positive")
        quantity = floor(self.config.target_notional / reference / self.config.lot_size)*self.config.lot_size
        if signal.direction < 0 and not self.config.allow_short and current_quantity is not None:
            quantity = min(quantity,max(0,current_quantity))
        if quantity < 1:
            return None, RiskDecision(False, "target_notional_below_one_share")
        order = OrderIntent(order_id, signal.symbol, signal.timestamp_ns, Side.BUY if signal.direction > 0 else Side.SELL, OrderType.MARKET, quantity)
        decision = self.risk.check(order, quote, current_notional, realized_pnl, now_ns, current_quantity=current_quantity, other_notional=other_notional)
        return (order if decision.approved else None), decision
