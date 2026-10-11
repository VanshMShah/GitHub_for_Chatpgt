"""Behavior-level regressions for daily readiness and durable paper execution."""
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
from src.ats.engine import ATSEngine
from src.ats.models import Forecast
from src.ats.market_session import MarketSession
from src.ats.gjr_garch import hansen_skewed_t_cdf,hansen_skewed_t_ppf
from src.ats.features import make_daily_features, SYMBOLS
from src.ats.store import Store
from src.ats.paper.account import PaperAccount
from src.ats.paper.types import OrderIntent,OrderType,Side,FillEvent,QuoteEvent
from src.ats.paper.risk import PaperRiskGuard
from src.ats.paper.simulator import DeterministicExecutionSimulator,SimulatorConfig
from src.ats.pqrnn_intraday import IntradayPQRNNLifecycle,IntradayPQRNNConfig,make_intraday_frame

@pytest.fixture
def engine(tmp_path,monkeypatch):
    from src.ats.config import settings
    monkeypatch.setattr('src.ats.engine.settings',SimpleNamespace(**{**settings.__dict__,'model_dir':tmp_path/'models'}))
    e=ATSEngine(Store(tmp_path/'state.sqlite3'))
    e.now_ist=lambda:datetime(2026,10,11,12,tzinfo=timezone.utc)
    return e

def add_daily(e,day,price):
    e.store.execute('INSERT INTO observations VALUES(?,?,?,?,?,?,?,?,?)',('^NSEI',day+'T00:00:00+00:00',day+'T12:00:00+00:00',price,price+1,price-1,price,1000,'test'))

def forecast_row(e,identity='f1',target='2026-10-12T10:00:00+00:00',reference=100):
    fc=Forecast(0,.6,-.03,-.01,0,.01,.03,.01,{})
    e._persist_forecast(identity,'2026-10-11T12:00:00+00:00',target,'1D',reference,fc,{},'calm',.8,{'experts':{'logistic':fc.__dict__},'factor_state':{'ret1':0,'ret5':0,'ret22':0,'rv_5':.01,'rv_22':.01}})

def order(identity='o1',qty=10,side=Side.BUY):
    return OrderIntent(identity,'NSE_EQ|TEST',1_000_000_000,side,OrderType.MARKET,qty)

def test_calendar_skips_weekend_and_holiday():
    cal=MarketSession()
    assert str(cal.next_session(datetime(2026,10,19).date()))=='2026-10-21'
    assert str(cal.previous_session(datetime(2026,10,11).date()))=='2026-10-09'

def test_cdf_matches_skew_t_quantiles():
    for skew in [-.5,0,.5]:
        for q in [.01,.05,.5,.95,.99]:
            assert hansen_skewed_t_cdf(hansen_skewed_t_ppf(q,8,skew),8,skew)==pytest.approx(q,abs=1e-8)

def test_future_target_not_settled_even_if_row_exists(engine):
    forecast_row(engine,target='2099-10-12T10:00:00+00:00')
    add_daily(engine,'2099-10-12',110)
    assert engine.settle()==0
    assert len(engine.store.query('SELECT * FROM outcomes'))==0

def test_settlement_is_once_and_updates_weights(engine):
    forecast_row(engine,target='2026-09-15T10:00:00+00:00')
    add_daily(engine,'2026-09-15',101)
    assert engine.settle()==1
    version=engine.policy.version
    assert engine.settle()==0
    assert engine.policy.version==version
    assert engine.ledger()[0]['state']=='SETTLED'
    assert engine.performance()['groups'][0]['n']==1

def test_settlement_uses_original_factor_state(engine):
    forecast_row(engine,target='2026-09-15T10:00:00+00:00')
    add_daily(engine,'2026-09-15',120)
    engine.settle()
    assert engine.residual.samples==1

def test_store_nested_transaction_rolls_back_all(tmp_path):
    s=Store(tmp_path/'s.db')
    with pytest.raises(RuntimeError):
        with s.transaction():
            s.set_state('a',1,'now');s.set_state('b',2,'now')
            raise RuntimeError('crash')
    assert s.get_state('a') is None and s.get_state('b') is None

def test_instrument_and_interval_never_mix(engine):
    row=('NSE_EQ|TEST',60,'2026-09-15T04:00:00+00:00','now',100,101,99,100,10,'test')
    engine.store.execute('INSERT INTO instrument_bars VALUES(?,?,?,?,?,?,?,?,?,?)',row)
    engine.store.execute('INSERT INTO instrument_bars VALUES(?,?,?,?,?,?,?,?,?,?)',('NSE_EQ|TEST',900,*row[2:]))
    frame=engine._load_intraday('NSE_EQ|TEST')
    assert len(frame)==1 and frame.attrs['instrument_key']=='NSE_EQ|TEST'
    assert engine._load_intraday('NSE_EQ|OTHER').empty

def test_account_fill_survives_restart_and_is_idempotent(tmp_path):
    path=tmp_path/'a.db';a=PaperAccount(Store(path),fee_bps=10)
    o=order();a.begin_session(o.symbol,'2026-09-15');assert a.reserve(o,'s1','f1')
    f=FillEvent(o.order_id,o.symbol,1_200_000_000,o.side,7,100,'depth')
    a.apply(o,f);a.apply(o,f)
    b=PaperAccount(Store(path),fee_bps=10);b.recover()
    assert b.position(o.symbol).quantity==7
    result=b.snapshot();assert result['fees']==pytest.approx(.7)
    assert result['orders'][0]['status']=='PARTIAL_CANCELLED'
    assert len(result['fills'])==1 and not b.reserve(o)

def test_interrupted_order_is_cancelled_not_resent(tmp_path):
    a=PaperAccount(Store(tmp_path/'a.db'));assert a.reserve(order())
    a.recover()
    assert a.snapshot()['orders'][0]['status']=='CANCELLED_ON_RESTART'
    assert not a.reserve(order())

def test_unrealized_losses_count_and_positions_carry_across_days(tmp_path):
    a=PaperAccount(Store(tmp_path/'a.db'));o=order();a.begin_session(o.symbol,'2026-09-15');a.reserve(o)
    a.apply(o,FillEvent(o.order_id,o.symbol,0,Side.BUY,10,100,'depth'));a.mark(o.symbol,90)
    assert a.daily_pnl(o.symbol)==-100
    a.begin_session(o.symbol,'2026-09-16');assert a.position(o.symbol).quantity==10
    assert a.daily_pnl(o.symbol)==0

def test_risk_reducing_exit_allowed_after_loss_limit():
    q=QuoteEvent('q','NSE_EQ|TEST',1_000_000_000,99,100,100,100)
    guard=PaperRiskGuard();o=order(qty=10,side=Side.SELL)
    assert guard.check(o,q,10000,-2000,q.timestamp_ns,current_quantity=10).approved
    assert not guard.check(order(),q,0,-2000,q.timestamp_ns,current_quantity=0).approved

def test_quote_source_freshness_not_receipt():
    q=QuoteEvent('q','NSE_EQ|TEST',1_000_000_000,99,100,100,100,source_timestamp_ns=1)
    assert PaperRiskGuard().check(order(),q,0,0,q.timestamp_ns).reason=='stale_quote'

def test_replay_shared_depth_and_all_or_nothing():
    q=QuoteEvent('q','NSE_EQ|TEST',1_200_000_000,99,100,5,5,ask_levels=((100,5),(101,5)))
    sim=DeterministicExecutionSimulator(SimulatorConfig(slippage_bps=0))
    assert sim.execute_market(order(qty=7),[q]).quantity==7
    assert sim.execute_market(order('o2',qty=7),[q]).quantity==3
    assert sim.execute_market(order('o3',qty=1),[q]) is None
    full=DeterministicExecutionSimulator(SimulatorConfig(slippage_bps=0,allow_partial=False))
    assert full.execute_market(order(qty=10),[q]).quantity==10

def test_backup_restores_database(tmp_path):
    s=Store(tmp_path/'s.db');s.set_state('test',{'a':1},'now');s.backup(tmp_path/'backup.db')
    assert Store(tmp_path/'backup.db').get_state('test')=={'a':1}

def raw_minutes(n=700):
    rng=np.random.default_rng(11);prices=100*np.exp(np.cumsum(rng.normal(0,.001,n)))
    x=pd.DataFrame({'event_time':pd.date_range('2026-09-15T04:00:00Z',periods=n,freq='60s'),'open':prices,'high':prices*1.001,'low':prices*.999,'close':prices,'volume':rng.integers(100,1000,n)})
    x.attrs['instrument_key']='NSE_EQ|TEST';return x

def test_intraday_sequence_includes_target_reference_features():
    life=IntradayPQRNNLifecycle.__new__(IntradayPQRNNLifecycle);life.cfg=IntradayPQRNNConfig()
    frame=make_intraday_frame(raw_minutes());X,y,_,_,mu,sd,_=life._sequences(frame)
    supervised,cols=life.make_supervised(frame,1)
    i=life.cfg.sequence_length-1
    np.testing.assert_allclose(X[0,-1],(supervised[list(cols)].iloc[i].to_numpy(dtype=np.float32)-mu)/sd,rtol=1e-5)
    assert y[0]==pytest.approx(supervised.iloc[i].target_ret)

def test_validation_and_holdout_cannot_affect_scaler():
    life=IntradayPQRNNLifecycle.__new__(IntradayPQRNNLifecycle);life.cfg=IntradayPQRNNConfig()
    frame=make_intraday_frame(raw_minutes());first=life._sequences(frame)
    supervised,_=life.make_supervised(frame,1)
    first_validation_time=supervised.iloc[life.cfg.sequence_length-1+life.validation_start]['event_time']
    altered=frame.copy();altered.loc[altered.event_time>=first_validation_time,list(life.cfg.feature_columns)]+=1000
    second=life._sequences(altered)
    np.testing.assert_allclose(first[4],second[4]);np.testing.assert_allclose(first[5],second[5])

def test_intraday_model_rejects_wrong_instrument(engine):
    torch=pytest.importorskip("torch")
    life=engine.pqrnn_intraday
    life.path.parent.mkdir(exist_ok=True)
    torch.save({'meta':{'schema_version':2,'promoted':True,'instrument_key':'NSE_EQ|OTHER'}},life.path)
    with pytest.raises(RuntimeError,match='instrument mismatch'):life.predict(raw_minutes())

def test_missing_exogenous_series_does_not_break_features():
    n=raw_minutes(100)
    result=make_daily_features(n,{'gold':pd.DataFrame()})
    assert 'gold' not in result

def test_daily_forecast_is_frozen_per_reference_and_target(engine):
    from test_engine import FakeProvider,seed
    engine.provider=FakeProvider();seed(engine)
    first=engine.forecast();second=engine.forecast()
    assert first==second
    assert engine.status()['forecasts']==1
    assert first['target_time']=='2026-10-12T10:00:00+00:00'

def test_daily_forecast_rejects_stale_reference(engine):
    from test_engine import FakeProvider,seed
    engine.provider=FakeProvider();seed(engine)
    engine.now_ist=lambda:datetime(2026,10,14,12,tzinfo=timezone.utc)
    with pytest.raises(RuntimeError,match='reference is stale'):engine.forecast()

def test_intraday_training_writes_auditable_challenger_without_bypassing_gates(engine,tmp_path):
    pytest.importorskip('torch')
    raw=raw_minutes();engine._load_intraday=lambda:raw
    engine.is_market_hours=lambda:False
    life=engine.pqrnn_intraday;life.path=tmp_path/'intraday.pt'
    life.cfg=IntradayPQRNNConfig(train_epochs=2)
    result=life.train()
    assert result['instrument_key']=='NSE_EQ|TEST'
    assert result['schema_version']==2
    assert result['status'] in ('promoted','rejected')
    assert result['holdout_rows']==100
    if result['status']=='promoted':
        prediction=life.predict(raw)
        assert 0<=prediction.p_up<=1
        assert prediction.q05<=prediction.q50<=prediction.q95
    else:
        assert not life.path.exists()
        assert life.path.with_name('pqrnn_intraday_challenger.pt').exists()

def test_live_signal_fill_restart_then_bar_settlement(engine,tmp_path):
    from src.ats.paper.live import LivePaperRunner,LivePaperConfig
    now=datetime(2026,9,15,5,2,tzinfo=timezone.utc)
    stamp=[int(now.timestamp()*1e9)]
    class Provider:
        instrument_key='NSE_EQ|TEST'
        def fetch_nifty_intraday(self,interval_minutes=1):
            return [{'event_time':'2026-09-15T05:01:00+00:00','open':100,'high':101,'low':99,'close':100,'volume':100,'source':'fixture'}]
        def fetch_full_quote(self):
            stamp[0]+=200_000_000
            return {'instrument_key':self.instrument_key,'timestamp':str(stamp[0]//1_000_000),'bids':((99.9,100),),'asks':((100,100),)}
    class Model:
        def predict(self,frame):
            assert frame.attrs['instrument_key']=='NSE_EQ|TEST'
            return Forecast(.001,.8,-.01,-.005,.001,.005,.01,.005,{'horizon':'60s','instrument_key':'NSE_EQ|TEST'})
    engine.pqrnn_intraday=Model()
    cfg=LivePaperConfig(enable=True,execution_instrument_key='NSE_EQ|TEST',audit_dir=str(tmp_path/'audit'),kill_switch_path=str(tmp_path/'kill'))
    runner=LivePaperRunner(engine,Provider(),cfg,now_fn=lambda:now,clock_ns=lambda:stamp[0])
    result=runner.tick();assert result['state']=='FILL',result
    assert runner.position.quantity==10
    again=runner.tick();assert again['state']=='HEALTHY_NO_NEW_BAR'
    restored=LivePaperRunner(engine,Provider(),cfg,now_fn=lambda:now,clock_ns=lambda:stamp[0]);assert restored.position.quantity==10
    assert len(restored.account.snapshot()['fills'])==1
    engine.store.execute('INSERT INTO instrument_bars VALUES(?,?,?,?,?,?,?,?,?,?)',('NSE_EQ|TEST',60,'2026-09-15T05:02:00+00:00','now',100,102,99,101,100,'fixture'))
    assert engine.settle()==1
    result=engine.performance()['groups'][0]
    assert result['symbol']=='NSE_EQ|TEST' and result['horizon']=='60s'


def test_derivative_partial_fill_respects_lot_size():
    q=QuoteEvent('q','NSE_EQ|TEST',1_200_000_000,99,100,7,7)
    sim=DeterministicExecutionSimulator(SimulatorConfig(slippage_bps=0,lot_size=5))
    assert sim.execute_market(order(qty=10),[q]).quantity==5


def test_overnight_gaps_reset_intraday_features():
    raw=raw_minutes(150)
    raw.loc[75:,'event_time']+=pd.Timedelta(days=1)
    features=make_intraday_frame(raw)
    assert pd.isna(features.loc[75,'ret'])
    assert pd.isna(features.loc[76,'rv48'])
