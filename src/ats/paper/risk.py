from __future__ import annotations

from dataclasses import dataclass

from .types import OrderIntent, QuoteEvent, Side


@dataclass(frozen=True)
class PaperRiskConfig:
    starting_capital: float = 100_000.0
    max_notional: float = 10_000.0
    max_order_notional: float = 2_000.0
    max_daily_loss: float = 1_000.0
    max_quote_age_ns: int = 500_000_000


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: str


class PaperRiskGuard:
    def __init__(self, config: PaperRiskConfig | None = None):
        self.config = config or PaperRiskConfig()

    def check(
        self,
        order: OrderIntent,
        quote: QuoteEvent | None,
        current_notional: float,
        realized_pnl: float,
        now_ns: int,
        current_quantity: int | None = None,
        unrealized_pnl: float = 0.0,
        other_notional: float = 0.0,
    ) -> RiskDecision:
        if quote is None:
            return RiskDecision(False, "missing_quote")
        if order.symbol != quote.symbol:
            return RiskDecision(False,"instrument_mismatch")
        stamp = quote.source_timestamp_ns if quote.source_timestamp_ns is not None else quote.timestamp_ns
        if now_ns-stamp < -100_000_000:
            return RiskDecision(False,"future_quote")
        if now_ns - stamp > self.config.max_quote_age_ns:
            return RiskDecision(False, "stale_quote")
        reference = quote.ask if order.side is Side.BUY else quote.bid
        notional = reference * order.quantity
        signed = order.quantity if order.side is Side.BUY else -order.quantity
        reducing = current_quantity is not None and current_quantity*signed < 0 and abs(signed)<=abs(current_quantity)
        projected = (abs(current_quantity+signed)*reference+other_notional if current_quantity is not None else current_notional+notional)
        if notional > self.config.max_order_notional and not reducing:
            return RiskDecision(False, "order_notional_limit")
        if projected > self.config.max_notional and not reducing:
            return RiskDecision(False, "portfolio_notional_limit")
        if realized_pnl + unrealized_pnl <= -abs(self.config.max_daily_loss) and not reducing:
            return RiskDecision(False, "daily_loss_limit")
        return RiskDecision(True, "approved")
