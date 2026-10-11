from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

from src.ats.research import evaluate
from src.ats.data import YahooChartProvider, Bar


class FakeProvider:
    def fetch(self, symbol, period="5y", interval="1d"):
        if interval != "1d":
            return []
        rng = np.random.default_rng(12)
        base = datetime(2018, 1, 1, tzinfo=timezone.utc)
        price = 10000.0
        out = []
        for i in range(520):
            ret = 0.0002 + rng.normal(0, 0.007)
            price *= np.exp(ret)
            ts = (base + timedelta(days=i)).isoformat()
            out.append(Bar(ts, price*0.999, price*1.001, price*0.998, price, 1000, "fake"))
        return out


def test_walk_forward_research_runs(monkeypatch):
    monkeypatch.setattr("src.ats.research.YahooChartProvider", FakeProvider)
    result = evaluate("5y", min_train=252, max_cases=5)
    assert result["n"] == 5
    assert set(result["models"]) == {"persistence", "logistic", "lightgbm_compatible", "arima_gjr_garch", "dma"}
    assert abs(sum(result["dma_final_weights"].values()) - 1.0) < 1e-9
