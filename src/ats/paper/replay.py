from __future__ import annotations

from dataclasses import dataclass

from .simulator import DeterministicExecutionSimulator
from .types import FillEvent, OrderIntent, Position, QuoteEvent


@dataclass(frozen=True)
class ReplayResult:
    fills: tuple[FillEvent, ...]
    position: Position


class PaperReplay:
    def __init__(self, simulator: DeterministicExecutionSimulator | None = None):
        self.simulator = simulator or DeterministicExecutionSimulator()

    def run(self, quotes: list[QuoteEvent], orders: list[OrderIntent]) -> ReplayResult:
        self.simulator = DeterministicExecutionSimulator(self.simulator.config)
        quotes = sorted(quotes, key=lambda q: q.timestamp_ns)
        orders = sorted(orders, key=lambda o: o.timestamp_ns)
        fills: list[FillEvent] = []
        position = Position()
        seen=set()
        for order in orders:
            if order.order_id in seen: continue
            seen.add(order.order_id)
            fill = self.simulator.execute_market(order, quotes)
            if fill is not None:
                fills.append(fill)
                position = position.apply(fill)
        return ReplayResult(tuple(fills), position)
