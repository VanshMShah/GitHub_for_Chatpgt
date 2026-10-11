from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from src.ats.store import Store
from src.ats.policy import DMAPolicy
from src.ats.models import Forecast

def test_store_round_trip(tmp_path: Path):
    s=Store(tmp_path/'ats.db'); now=datetime.now(timezone.utc).isoformat()
    s.set_state('x',{'a':1},now); assert s.get_state('x')=={'a':1}

def test_policy_updates_and_serializes():
    p=DMAPolicy(); fs={k:Forecast(0.0,0.5,-.02,-.01,0,.01,.02,.01,{}) for k in p.weights}
    old=p.version; p.observe(-.01,fs); state=p.state()
    assert state['version']==old+1 and all(np.isfinite(v) for v in state['loss_ema'].values())

def test_policy_weights_adapt():
    p=DMAPolicy()
    fs={
        "persistence": Forecast(0.0,0.5,-.02,-.01,0,.01,.02,.01,{}),
        "logistic": Forecast(0.0,0.95,-.02,-.01,0,.01,.02,.01,{}),
        "lightgbm_compatible": Forecast(0.0,0.95,-.02,-.01,0,.01,.02,.01,{}),
        "arima_gjr_garch": Forecast(0.0,0.5,-.02,-.01,0,.01,.02,.01,{}),
    }
    before=dict(p.weights)
    p.observe(0.01,fs)
    assert p.weights != before
    assert abs(sum(p.weights.values())-1.0)<1e-9
