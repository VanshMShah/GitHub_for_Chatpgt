from src.ats.live import UpstoxLiveProvider


def test_upstox_full_quote_normalizes_top_five_depth(monkeypatch):
    payload = {
        "status": "success",
        "data": {
            "NSE_EQ:TEST": {
                "timestamp": "2026-09-15T10:00:00+05:30",
                "last_price": 100.5,
                "depth": {
                    "buy": [
                        {"price": 100.4, "quantity": 50, "orders": 3},
                        {"price": 100.3, "quantity": 100, "orders": 4},
                    ],
                    "sell": [
                        {"price": 100.6, "quantity": 60, "orders": 5},
                        {"price": 100.7, "quantity": 80, "orders": 6},
                    ],
                },
                "ohlc": {"close": 100.0},
            }
        },
    }
    class FakeResponse:
        pass
    monkeypatch.setattr("src.ats.live.fetch_json", lambda req, timeout=8: payload)
    provider = UpstoxLiveProvider("token", "NSE_EQ|TEST")
    result = provider.fetch_full_quote()
    assert result["instrument_key"] == "NSE_EQ|TEST"
    assert result["bids"] == ((100.4, 50), (100.3, 100))
    assert result["asks"] == ((100.6, 60), (100.7, 80))
