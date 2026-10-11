from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass(frozen=True)
class QuoteEvent:
    event_id: str
    symbol: str
    timestamp_ns: int
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    bid_levels: tuple[tuple[float, int], ...] = ()
    ask_levels: tuple[tuple[float, int], ...] = ()
    source_timestamp_ns: int | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.bid) or not isfinite(self.ask) or self.bid <= 0 or self.ask <= 0 or self.ask < self.bid:
            raise ValueError("Invalid bid/ask")
        if self.bid_size < 0 or self.ask_size < 0:
            raise ValueError("Quote sizes must be non-negative")
        for price, size in (*self.bid_levels, *self.ask_levels):
            if not isfinite(price) or price <= 0 or size < 0:
                raise ValueError("Depth levels must contain positive prices and non-negative sizes")


@dataclass(frozen=True)
class OrderIntent:
    order_id: str
    symbol: str
    timestamp_ns: int
    side: Side
    order_type: OrderType
    quantity: int
    limit_price: float | None = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.order_type is OrderType.LIMIT and (self.limit_price is None or self.limit_price <= 0):
            raise ValueError("limit orders require a positive limit_price")


@dataclass(frozen=True)
class FillEvent:
    order_id: str
    symbol: str
    timestamp_ns: int
    side: Side
    quantity: int
    price: float
    reason: str


@dataclass(frozen=True)
class Position:
    quantity: int = 0
    average_price: float = 0.0
    realized_pnl: float = 0.0

    def apply(self, fill: FillEvent) -> "Position":
        signed = fill.quantity if fill.side is Side.BUY else -fill.quantity
        old_qty = self.quantity
        new_qty = old_qty + signed
        if old_qty == 0 or (old_qty > 0 and signed > 0) or (old_qty < 0 and signed < 0):
            total_notional = abs(old_qty) * self.average_price + abs(signed) * fill.price
            return Position(new_qty, total_notional / abs(new_qty), self.realized_pnl)

        closing = min(abs(old_qty), abs(signed))
        direction = 1 if old_qty > 0 else -1
        realized = self.realized_pnl + closing * (fill.price - self.average_price) * direction
        if new_qty == 0:
            return Position(0, 0.0, realized)
        if abs(signed) < abs(old_qty):
            return Position(new_qty, self.average_price, realized)
        return Position(new_qty, fill.price, realized)
