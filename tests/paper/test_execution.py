from src.ats.paper.replay import PaperReplay
from src.ats.paper.simulator import DeterministicExecutionSimulator, SimulatorConfig
from src.ats.paper.types import OrderIntent, OrderType, QuoteEvent, Side


def _quotes():
    return [
        QuoteEvent("q1", "TEST", 1_000_000_000, 99.0, 100.0, 10, 5),
        QuoteEvent("q2", "TEST", 1_200_000_000, 101.0, 102.0, 10, 5),
    ]


def test_market_buy_uses_post_latency_ask_not_order_time_quote():
    sim = DeterministicExecutionSimulator(SimulatorConfig(latency_ns=150_000_000, slippage_bps=0))
    order = OrderIntent("o1", "TEST", 1_000_000_000, Side.BUY, OrderType.MARKET, 1)
    fill = sim.execute_market(order, _quotes())
    assert fill is not None
    assert fill.timestamp_ns == 1_200_000_000
    assert fill.price == 102.0


def test_partial_fill_is_conservative():
    sim = DeterministicExecutionSimulator(SimulatorConfig(latency_ns=150_000_000, slippage_bps=0))
    order = OrderIntent("o1", "TEST", 1_000_000_000, Side.BUY, OrderType.MARKET, 9)
    fill = sim.execute_market(order, _quotes())
    assert fill is not None
    assert fill.quantity == 5


def test_replay_is_deterministic():
    order = OrderIntent("o1", "TEST", 1_000_000_000, Side.BUY, OrderType.MARKET, 2)
    replay = PaperReplay()
    a = replay.run(_quotes(), [order])
    b = replay.run(_quotes(), [order])
    assert a == b
