from src.ats.models import Forecast
from src.ats.paper.bridge import ForecastOrderBridge, SignalConfig
from src.ats.paper.risk import PaperRiskConfig, PaperRiskGuard
from src.ats.paper.types import QuoteEvent, Side
from src.ats.paper.replay import PaperReplay


def quote() -> QuoteEvent:
    return QuoteEvent("q1", "TEST", 1_000_000_000, 99.0, 101.0, 100, 100)


def test_pqrnn_forecast_becomes_risk_checked_buy_order():
    bridge = ForecastOrderBridge(
        SignalConfig(expected_horizon="1D", buy_threshold=0.55, target_notional=1_000.0),
        PaperRiskGuard(PaperRiskConfig(max_order_notional=1_100.0, max_notional=2_000.0)),
    )
    forecast = Forecast(0.01, 0.70, -0.01, -0.002, 0.01, 0.02, 0.04, 0.02, {"expert": "pqrnn", "horizon": "1D"})
    signal = bridge.signal(forecast, "TEST", 1_000_000_000, "s1")
    order, decision = bridge.order(signal, quote(), "o1", 1_000_000_000)
    assert signal.direction == 1
    assert decision.approved and order is not None
    assert order.side is Side.BUY


def test_daily_pqrnn_cannot_be_silently_used_as_intraday_signal():
    bridge = ForecastOrderBridge(SignalConfig(expected_horizon="1m"))
    forecast = Forecast(0.01, 0.70, -0.01, -0.002, 0.01, 0.02, 0.04, 0.02, {"expert": "pqrnn", "horizon": "1D"})
    try:
        bridge.signal(forecast, "TEST", 1_000_000_000, "s1")
    except ValueError as exc:
        assert "forecast horizon" in str(exc)
    else:
        raise AssertionError("horizon mismatch must be rejected")


def test_forecast_to_fill_to_position_replay_is_end_to_end():
    bridge = ForecastOrderBridge(
        SignalConfig(expected_horizon="1D", target_notional=1_000.0),
        PaperRiskGuard(PaperRiskConfig(max_order_notional=1_100.0, max_notional=2_000.0)),
    )
    forecast = Forecast(0.01, 0.70, -0.01, -0.002, 0.01, 0.02, 0.04, 0.02, {"expert": "pqrnn", "horizon": "1D"})
    signal = bridge.signal(forecast, "TEST", 1_000_000_000, "s1")
    order, decision = bridge.order(signal, quote(), "o1", 1_000_000_000)
    assert decision.approved and order is not None
    result = PaperReplay().run(
        [quote(), QuoteEvent("q2", "TEST", 1_150_000_000, 100.0, 102.0, 100, 100)],
        [order],
    )
    assert len(result.fills) == 1
    assert result.position.quantity > 0


def test_depth_simulator_consumes_multiple_levels_for_market_order():
    from src.ats.paper.simulator import DeterministicExecutionSimulator, SimulatorConfig
    from src.ats.paper.types import OrderIntent, OrderType, QuoteEvent, Side
    q = QuoteEvent(
        "q-depth", "NSE_EQ|TEST", 200_000_000,
        100.0, 100.1, 50, 50,
        bid_levels=((100.0, 50), (99.9, 100)),
        ask_levels=((100.1, 50), (100.2, 100)),
    )
    order = OrderIntent("o-depth", "NSE_EQ|TEST", 50_000_000, Side.BUY, OrderType.MARKET, 100)
    sim = DeterministicExecutionSimulator(SimulatorConfig(latency_ns=100_000_000, slippage_bps=0.0))
    fill = sim.execute_market(order, [q])
    assert fill is not None
    assert fill.quantity == 100
    assert abs(fill.price - 100.15) < 1e-9
