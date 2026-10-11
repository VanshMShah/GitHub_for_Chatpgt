"""Offline synthetic demonstration, isolated from real research and paper accounts."""
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from src.ats.engine import ATSEngine
from src.ats.features import SYMBOLS
from src.ats.market_session import MarketSession,IST
from src.ats.store import Store
from src.ats.config import settings

def seed_demo():
    store=Store(settings.db_path)
    if store.get_state('demo_ready'):return
    cal=MarketSession();now=datetime.now(IST);last=now.date()
    if not cal.is_market_day(now) or now<cal.close_at(last)+pd.Timedelta(minutes=settings.daily_publication_lag_minutes):last=cal.previous_session(last)
    days=pd.bdate_range(end=str(last),periods=520,tz='UTC')
    rng=np.random.default_rng(42)
    for name,symbol in SYMBOLS.items():
        level=25000 if name=='nifty' else (20 if name in ('vix','india_vix') else 100)
        prices=level*np.exp(np.cumsum(rng.normal(0,.008,len(days))))
        rows=[]
        for day,price in zip(days,prices):
            stamp=day.isoformat();available=(day+pd.Timedelta(hours=12)).isoformat()
            rows.append((symbol,stamp,available,price*.999,price*1.002,price*.998,price,1000000,'DEMO_SYNTHETIC') if name=='nifty' else (symbol,stamp,available,price,'DEMO_SYNTHETIC'))
        sql='INSERT OR REPLACE INTO observations VALUES(?,?,?,?,?,?,?,?,?)' if name=='nifty' else 'INSERT OR REPLACE INTO exogenous VALUES(?,?,?,?,?)'
        store.executemany(sql,rows)
    stamp=datetime.now(timezone.utc).isoformat()
    latest=store.query('SELECT event_time,close FROM observations ORDER BY event_time DESC LIMIT 1')[0]
    store.execute('INSERT OR REPLACE INTO live_quotes VALUES(?,?,?,?,?,?)',('^NSEI',cal.close_at(last).astimezone(timezone.utc).isoformat(),stamp,latest[1],0,'DEMO_SYNTHETIC'))
    engine=ATSEngine(store);engine.train();engine.forecast()
    store.set_state('demo_ready',True,stamp)
    store.close()
