from __future__ import annotations

from dataclasses import dataclass

from .types import FillEvent, OrderIntent, OrderType, QuoteEvent, Side


@dataclass(frozen=True)
class SimulatorConfig:
    latency_ns: int = 150_000_000
    slippage_bps: float = 1.0
    max_quote_age_ns: int = 500_000_000
    allow_partial: bool = True
    lot_size: int = 1


class DeterministicExecutionSimulator:
    """Replay-safe execution model using future quotes after order latency."""

    def __init__(self, config: SimulatorConfig | None = None):
        self.config = config or SimulatorConfig()
        self.used = {}
        self.completed = {}
        if min(self.config.latency_ns,self.config.slippage_bps,self.config.max_quote_age_ns)<0:
            raise ValueError("Simulator parameters must be nonnegative")

    def execute_market(self, order: OrderIntent, quotes: list[QuoteEvent]) -> FillEvent | None:
        if order.order_id in self.completed:
            return self.completed[order.order_id]
        if order.order_type is not OrderType.MARKET:
            raise ValueError("Only market IOC orders are supported")
        if self.config.lot_size<1 or order.quantity % self.config.lot_size:
            raise ValueError("Order quantity does not match lot size")
        arrival_ns = order.timestamp_ns + self.config.latency_ns
        quote = self._quote_at_or_after(order.symbol, arrival_ns, quotes)
        if quote is None or quote.timestamp_ns - order.timestamp_ns > self.config.max_quote_age_ns:
            return None
        levels = quote.ask_levels if order.side is Side.BUY else quote.bid_levels
        if not levels:
            levels = ((quote.ask, quote.ask_size),) if order.side is Side.BUY else ((quote.bid, quote.bid_size),)
        source_ns = quote.source_timestamp_ns if quote.source_timestamp_ns is not None else quote.timestamp_ns
        if quote.timestamp_ns-source_ns > self.config.max_quote_age_ns or source_ns-quote.timestamp_ns > 100_000_000:
            return None
        levels = sorted(levels, reverse=order.side is Side.SELL)
        consumed = self.used.setdefault((quote.symbol,quote.event_id,order.side.value),{})
        total_available=sum(max(0,size-consumed.get(price,0)) for price,size in levels)
        if not self.config.allow_partial and total_available<order.quantity:
            return None
        changes={}
        remaining = min(order.quantity,total_available//self.config.lot_size*self.config.lot_size)
        filled = 0
        notional = 0.0
        for base, available in levels:
            if remaining <= 0:
                break
            take = min(remaining, max(0,available-consumed.get(base,0)))
            if take <= 0:
                continue
            slip = base * self.config.slippage_bps / 10_000.0
            exec_price = base + slip if order.side is Side.BUY else base - slip
            changes[base]=changes.get(base,0)+take
            filled += take
            remaining -= take
            notional += take * exec_price
        if filled <= 0 or (not self.config.allow_partial and filled != order.quantity):
            return None
        for price,qty in changes.items(): consumed[price]=consumed.get(price,0)+qty
        result=FillEvent(order.order_id, order.symbol, quote.timestamp_ns, order.side, filled, notional / filled, "market_depth")
        self.completed[order.order_id]=result
        return result

    @staticmethod
    def _quote_at_or_after(symbol: str, timestamp_ns: int, quotes: list[QuoteEvent]) -> QuoteEvent | None:
        for quote in quotes:
            if quote.symbol == symbol and quote.timestamp_ns >= timestamp_ns:
                return quote
        return None
