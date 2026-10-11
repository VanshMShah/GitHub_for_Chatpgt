import pytest
pytest.importorskip("torch", reason="Optional neural tests require torch")
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
from src.ats.store import Store
from src.ats.engine import ATSEngine
from src.ats.data import Bar
from src.ats.features import SYMBOLS
class F:
 def fetch(self,symbol,period='5d',interval='15m'):
  n=500;base=datetime(2024,1,1,tzinfo=timezone.utc);r=np.random.default_rng(4);p=10000*np.exp(np.cumsum(r.normal(0,.002,n)));return [Bar((base+timedelta(minutes=15*i)).isoformat(),x*.999,x*1.001,x*.998,x,1000,'f') for i,x in enumerate(p)]
def test_intraday_15m_data_is_rejected_for_60s_pqrnn(tmp_path:Path):
 e=ATSEngine(Store(tmp_path/'a.db'));e.provider=F();e.ingest_intraday()
 try:
  e.train_intraday()
 except RuntimeError as exc:
  assert "approximately 60-second bars" in str(exc)
 else:
  raise AssertionError("15-minute data must not train the 60-second PQRNN")
