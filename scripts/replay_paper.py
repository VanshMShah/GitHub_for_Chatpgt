from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ats.paper.replay import PaperReplay
from src.ats.paper.types import OrderIntent, OrderType, QuoteEvent, Side


def read_quotes(path: str) -> list[QuoteEvent]:
    out = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out.append(QuoteEvent(
                row["event_id"], row["symbol"], int(row["timestamp_ns"]),
                float(row["bid"]), float(row["ask"]), int(row["bid_size"]), int(row["ask_size"]),
            ))
    return out


def read_orders(path: str) -> list[OrderIntent]:
    out = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out.append(OrderIntent(
                row["order_id"], row["symbol"], int(row["timestamp_ns"]),
                Side(row["side"]), OrderType.MARKET, int(row["quantity"]),
            ))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic ATS paper-trading replay")
    parser.add_argument("--quotes", required=True, help="CSV quote event file")
    parser.add_argument("--orders", required=True, help="CSV order intent file")
    args = parser.parse_args()
    result = PaperReplay().run(read_quotes(args.quotes), read_orders(args.orders))
    print(f"fills={len(result.fills)} position={result.position.quantity} realized_pnl={result.position.realized_pnl:.6f}")


if __name__ == "__main__":
    main()
