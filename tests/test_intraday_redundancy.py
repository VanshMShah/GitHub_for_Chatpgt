from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from src.ats.store import Store
from src.ats.engine import ATSEngine
from src.ats.data import Bar
from src.ats.features import SYMBOLS

class ShortThenLong:
    def __init__(self): self.calls=[]
    def fetch(self, symbol, period='10d', interval='15m'):
        self.calls.append((symbol, period, interval))
        n = 160 if period == '10d' else 260
        base=datetime(2024,1,1,tzinfo=timezone.utc)
        rng=np.random.default_rng(5)
        p=10000*np.exp(np.cumsum(rng.normal(0,.002,n)))
        return [Bar((base+timedelta(minutes=15*i)).isoformat(),x*.999,x*1.001,x*.998,x,1000,'f') for i,x in enumerate(p)]

def test_intraday_progressive_backfill(tmp_path: Path):
    e=ATSEngine(Store(tmp_path/'a.db'))
    e.provider=ShortThenLong()
    r=e.ingest_intraday('10d','15m')
    assert r['nifty'] >= 200
    assert e.provider.calls[0][1] == '10d'
    assert e.provider.calls[1][1] == '20d'
