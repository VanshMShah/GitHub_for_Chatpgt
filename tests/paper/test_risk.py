from src.ats.paper.risk import PaperRiskConfig, PaperRiskGuard
from src.ats.paper.types import OrderIntent, OrderType, QuoteEvent, Side


def _order(qty=10):
    return OrderIntent("o1", "TEST", 1_000, Side.BUY, OrderType.MARKET, qty)


def _quote():
    return QuoteEvent("q1", "TEST", 900, 99.0, 100.0, 100, 100)


def test_risk_approves_valid_order():
    guard = PaperRiskGuard(PaperRiskConfig(max_order_notional=2_000))
    assert guard.check(_order(), _quote(), 0, 0, 1_000).approved


def test_risk_rejects_stale_quote():
    guard = PaperRiskGuard(PaperRiskConfig(max_quote_age_ns=50))
    decision = guard.check(_order(), _quote(), 0, 0, 1_000)
    assert not decision.approved
    assert decision.reason == "stale_quote"


def test_risk_rejects_daily_loss():
    guard = PaperRiskGuard(PaperRiskConfig(max_daily_loss=100))
    decision = guard.check(_order(), _quote(), 0, -100, 1_000)
    assert not decision.approved
    assert decision.reason == "daily_loss_limit"
