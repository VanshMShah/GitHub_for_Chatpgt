from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from src.ats.store import Store
from src.ats.engine import ATSEngine
from src.ats.data import Bar
from src.ats.features import SYMBOLS

class FakeProvider:
    def fetch(self, symbol, period='2y', interval='1d'):
        n=420; timestamps=__import__("pandas").bdate_range(end="2026-10-09",periods=n,tz="UTC"); rng=np.random.default_rng(7)
        rets=rng.normal(0.0003,0.012,n); rets[170:180]+=rng.normal(-0.001,0.025,10)
        prices=10000*np.exp(np.cumsum(rets)); out=[]
        for i,p in enumerate(prices):
            dt=timestamps[i].isoformat(); out.append(Bar(dt,p*.999,p*1.002,p*.998,p,1000000,'fake'))
        return out

def seed(e):
    now=datetime.now(timezone.utc).isoformat()
    for name,symbol in SYMBOLS.items():
        bars=e.provider.fetch(symbol)
        if name=='nifty': e.store.executemany('INSERT INTO observations VALUES(?,?,?,?,?,?,?,?,?)',[(symbol,b.event_time,now,b.open,b.high,b.low,b.close,b.volume,b.source) for b in bars])
        else: e.store.executemany('INSERT INTO exogenous VALUES(?,?,?,?,?)',[(symbol,b.event_time,now,b.close,b.source) for b in bars])

def test_engine_training_and_forecast(tmp_path: Path):
    e=ATSEngine(Store(tmp_path/'ats.db')); e.provider=FakeProvider(); seed(e)
    e.now_ist=lambda:datetime(2026,10,11,12,tzinfo=timezone.utc).astimezone(__import__("zoneinfo").ZoneInfo("Asia/Kolkata"))
    r=e.train(); assert r['rows']>=220
    out=e.forecast(); assert 0 < out['forecast']['p_up'] < 1
    assert out['forecast']['q05']<=out['forecast']['q50']<=out['forecast']['q95']
    assert e.status()['forecasts']==1

def test_live_quote_is_separate_from_daily_observations(tmp_path: Path):
    e=ATSEngine(Store(tmp_path/'ats.db'))
    # Exercise the storage contract directly, avoiding network.
    e.store.execute("INSERT INTO live_quotes VALUES(?,?,?,?,?,?)", (SYMBOLS['nifty'], '2026-01-09T10:00:00+00:00', '2026-01-09T10:00:01+00:00', 24000.0, 123.0, 'FakeLive'))
    assert e.store.query("SELECT COUNT(*) FROM observations")[0][0] == 0
    assert e.store.query("SELECT COUNT(*) FROM live_quotes")[0][0] == 1
