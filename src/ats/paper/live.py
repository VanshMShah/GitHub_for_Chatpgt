from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
import pandas as pd
from src.ats.market_session import MarketSession
from src.ats.paper.account import PaperAccount
from src.ats.paper.audit import AuditLogger
from src.ats.paper.bridge import ForecastOrderBridge, SignalConfig
from src.ats.paper.risk import PaperRiskConfig, PaperRiskGuard
from src.ats.paper.simulator import DeterministicExecutionSimulator, SimulatorConfig
from src.ats.paper.types import QuoteEvent, OrderIntent, OrderType, Side

UTC=timezone.utc
@dataclass(frozen=True)
class LivePaperConfig:
    interval_seconds:int=60
    stale_quote_ns:int=2_000_000_000
    target_notional:float=1000.0
    kill_switch_path:str='data/PAPER_KILL'
    audit_dir:str='data/live_paper'
    enable:bool=False
    execution_instrument_key:str=''
    lot_size:int=1
    lot_size_confirmed:bool=False
    allow_short:bool=False
    fee_bps:float=0.0
    fixed_fee:float=0.0
    flatten_minutes:int=5
    max_order_notional:float=2000.0
    max_notional:float=10000.0
    max_daily_loss:float=1000.0

class LivePaperRunner:
    """REST depth paper-only evaluator; durable IOC orders and position recovery."""
    def __init__(self,engine,provider,config=None,now_fn:Callable[[],datetime]|None=None,clock_ns=None):
        self.engine,self.provider=engine,provider
        self.cfg=config or LivePaperConfig()
        self.execution_instrument_key=self.cfg.execution_instrument_key or getattr(provider,'instrument_key','')
        self.now_fn=now_fn or (lambda:datetime.now(UTC))
        self.clock_ns=clock_ns or time.time_ns
        self.session=MarketSession()
        self.account=PaperAccount(engine.store,self.cfg.fee_bps,self.cfg.fixed_fee)
        self.account.recover()
        self.position=self.account.position(self.execution_instrument_key)
        self.quote_history=[]
        self._session_date=None
        self._audit=None
        self.risk=PaperRiskGuard(PaperRiskConfig(max_quote_age_ns=self.cfg.stale_quote_ns,max_order_notional=self.cfg.max_order_notional,max_notional=self.cfg.max_notional,max_daily_loss=self.cfg.max_daily_loss))
        self.bridge=ForecastOrderBridge(SignalConfig(expected_horizon='60s',target_notional=self.cfg.target_notional,lot_size=self.cfg.lot_size,allow_short=self.cfg.allow_short),self.risk)
        self.simulator=DeterministicExecutionSimulator(SimulatorConfig(max_quote_age_ns=self.cfg.stale_quote_ns,lot_size=self.cfg.lot_size))

    @property
    def kill_switch_active(self): return Path(self.cfg.kill_switch_path).exists()

    def _heartbeat(self,state,**extra):
        payload={'timestamp_utc':self.now_fn().astimezone(UTC).isoformat(),'state':state,'instrument':self.execution_instrument_key,**extra}
        self.engine.store.set_state('live_paper_heartbeat',payload,payload['timestamp_utc'])
        if self._audit: self._audit.write('HEARTBEAT',**payload)
        return payload

    def _ensure_session(self,now):
        date=self.session.session_date(now)
        if date!=self._session_date:
            self._session_date=date
            self.quote_history.clear()
            self._audit=AuditLogger(Path(self.cfg.audit_dir)/f'paper_{date}.jsonl') if date else None
            if date:
                self.account.begin_session(self.execution_instrument_key,date)
                self.position=self.account.position(self.execution_instrument_key)
                self._audit.write('SESSION_START',session_date=date,position=self.position.__dict__,policy='flatten_before_close; retain and flag failed exits')
        return bool(date)

    def _validate_execution_instrument(self):
        key=self.execution_instrument_key
        if not key or not key.startswith(('NSE_EQ|','NSE_FO|')):
            raise RuntimeError('LIVE_DATA/PAPER requires a tradeable NSE instrument with genuine depth; NSE_INDEX cannot be executed')
        if getattr(self.provider,'instrument_key',None)!=key:
            raise RuntimeError('Depth provider instrument mismatch')
        if key.startswith('NSE_FO|') and not self.cfg.lot_size_confirmed:
            raise RuntimeError('Set ATS_PAPER_LOT_SIZE from the instrument contract before using derivatives')
        if self.cfg.lot_size<1: raise RuntimeError('Lot size must be positive')

    def _genuine_quote(self):
        self._validate_execution_instrument()
        snapshot=self.provider.fetch_full_quote()
        if snapshot.get('instrument_key')!=self.execution_instrument_key: raise RuntimeError('Quote instrument mismatch')
        value=str(snapshot['timestamp'])
        if value.replace('.','',1).isdigit(): source=int(float(value)*1_000_000)
        else:
            ts=pd.Timestamp(value)
            if ts.tzinfo is None: raise RuntimeError('Depth source timestamp has no timezone')
            source=ts.value
        receipt=self.clock_ns()
        age=receipt-source
        if age>self.cfg.stale_quote_ns or age < -100_000_000:
            raise RuntimeError('Exchange depth timestamp is stale or in the future')
        bids=tuple(sorted(snapshot['bids'],reverse=True)); asks=tuple(sorted(snapshot['asks']))
        return QuoteEvent(hashlib.sha1(f"{self.execution_instrument_key}:{source}:{bids}:{asks}".encode()).hexdigest()[:20],self.execution_instrument_key,receipt,bids[0][0],asks[0][0],bids[0][1],asks[0][1],bids,asks,source)

    def _closed_rows(self):
        now=pd.Timestamp(self.now_fn())
        out=[]
        for r in self.provider.fetch_nifty_intraday(1):
            ts=pd.Timestamp(r['event_time'])
            if ts.tzinfo is None: raise RuntimeError('Bar timestamp has no timezone')
            ts=ts.tz_convert('UTC')
            if ts+pd.Timedelta(seconds=60)<=now: out.append({**r,'event_time':ts.isoformat()})
        return sorted(out,key=lambda r:r['event_time'])

    def _ingest(self,rows):
        new=[]
        for r in rows:
            values=[float(r[k]) for k in ('open','high','low','close','volume')]
            import math
            if not all(math.isfinite(v) for v in values) or min(values[:4])<=0 or values[4]<0:
                raise RuntimeError('Invalid market bar')
            exists=self.engine.store.query('SELECT 1 FROM instrument_bars WHERE symbol=? AND interval_seconds=60 AND event_time=?',(self.execution_instrument_key,r['event_time']))
            if exists: continue
            stamp=self.now_fn().astimezone(UTC).isoformat()
            self.engine.store.execute('INSERT OR IGNORE INTO instrument_bars VALUES(?,?,?,?,?,?,?,?,?,?)',(self.execution_instrument_key,60,r['event_time'],stamp,*values,r['source']))
            # Retain legacy observation contract for old tools, but model reads interval-specific table.
            self.engine.store.execute('INSERT OR IGNORE INTO intraday_observations VALUES(?,?,?,?,?,?,?,?,?)',(self.execution_instrument_key,r['event_time'],stamp,*values,r['source']))
            new.append(r)
            if self._audit: self._audit.write('MARKET_BAR',**r)
        return new

    def _execute_live_market(self,order,order_timestamp_ns=None):
        time.sleep(self.simulator.config.latency_ns/1e9)
        if self.kill_switch_active:
            self.account.finish(order.order_id,'CANCELLED','kill switch activated before fill')
            return None
        quote=self._genuine_quote()
        position=self.account.position(order.symbol)
        snapshot=self.account.snapshot()
        other=max(0,snapshot['notional']-abs(position.quantity)*(quote.bid if position.quantity>=0 else quote.ask))
        decision=self.risk.check(order,quote,0,self.account.daily_pnl(order.symbol),self.clock_ns(),current_quantity=position.quantity,other_notional=other)
        if not decision.approved:
            self.account.finish(order.order_id,'CANCELLED',decision.reason)
            return None
        self.quote_history.append(quote)
        self.quote_history=self.quote_history[-100:]
        if self._audit: self._audit.write('EXECUTION_DEPTH',quote=quote.__dict__)
        return self.simulator.execute_market(order,[quote])

    def _execute(self,order,signal_id=None,forecast_id=None):
        if not self.account.reserve(order,signal_id,forecast_id): return self._heartbeat('DUPLICATE_ORDER',order_id=order.order_id)
        try: fill=self._execute_live_market(order)
        except Exception as exc:
            self.account.finish(order.order_id,'CANCELLED',str(exc))
            return self._heartbeat('ORDER_NO_FILL',reason=str(exc),order_id=order.order_id)
        if fill is None:
            self.account.finish(order.order_id,'CANCELLED','No executable post-latency quote; IOC cancelled')
            return self._heartbeat('ORDER_NO_FILL',order_id=order.order_id)
        self.position=self.account.apply(order,fill)
        if self._audit: self._audit.write('FILL',fill=fill.__dict__,position=self.position.__dict__,forecast_id=forecast_id)
        return self._heartbeat('FILL',order_id=order.order_id,fill=fill.__dict__,position=self.position.__dict__)

    def flatten(self):
        now=self.now_fn()
        if not self.cfg.enable: return self._heartbeat('DISABLED')
        if self.kill_switch_active: return self._heartbeat('HALTED_KILL_SWITCH')
        if not self.session.is_open(now): return self._heartbeat('EXIT_PENDING',reason='Market closed; position retained')
        pos=self.account.position(self.execution_instrument_key)
        if not pos.quantity: return self._heartbeat('FLAT')
        quote=self._genuine_quote()
        self.account.mark(quote.symbol,quote.bid if pos.quantity>0 else quote.ask)
        identity=f"exit:{quote.symbol}:{pos.quantity}:{quote.event_id}"
        order=OrderIntent(hashlib.sha1(identity.encode()).hexdigest()[:20],quote.symbol,self.clock_ns(),Side.SELL if pos.quantity>0 else Side.BUY,OrderType.MARKET,abs(pos.quantity))
        return self._execute(order,'session_exit')

    def tick(self):
        from filelock import FileLock
        with FileLock(str(self.engine.store.path)+".workflow.lock",timeout=600):
            return self._tick()

    def _tick(self):
        now=self.now_fn().astimezone(UTC)
        if not self._ensure_session(now): return self._heartbeat('OFF_MARKET')
        if not self.session.is_open(now):
            return self._heartbeat('EXIT_PENDING' if self.position.quantity else 'SESSION_CLOSED',reason='Position retained until executable exit' if self.position.quantity else '')
        if not self.cfg.enable: return self._heartbeat('DISABLED')
        if self.kill_switch_active: return self._heartbeat('HALTED_KILL_SWITCH')
        if now >= self.session.close_at(now.astimezone(__import__('zoneinfo').ZoneInfo('Asia/Kolkata')).date()).astimezone(UTC)-pd.Timedelta(minutes=self.cfg.flatten_minutes):
            return self.flatten()
        rows=self._closed_rows(); new=self._ingest(rows)
        if not rows: return self._heartbeat('HEALTHY_NO_NEW_BAR')
        try: self._validate_execution_instrument()
        except Exception as exc: return self._heartbeat('BLOCKED_NO_GENUINE_DEPTH',reason=str(exc))
        bar=rows[-1]
        if pd.Timestamp(now)-(pd.Timestamp(bar['event_time'])+pd.Timedelta(seconds=60))>pd.Timedelta(seconds=90):
            return self._heartbeat('NO_ORDER',reason='Latest closed bar is stale')
        try: quote=self._genuine_quote()
        except Exception as exc: return self._heartbeat('BLOCKED_NO_GENUINE_DEPTH',reason=str(exc))
        position=self.account.position(quote.symbol)
        self.account.mark(quote.symbol,quote.bid if position.quantity>=0 else quote.ask)
        if self.account.daily_pnl(quote.symbol)<=-self.cfg.max_daily_loss and position.quantity: return self.flatten()
        frame=self.engine._load_intraday(symbol=quote.symbol)
        try: forecast=self.engine.pqrnn_intraday.predict(frame)
        except Exception as exc: return self._heartbeat('WARMUP',reason=str(exc),bars=len(frame))
        if forecast is None: return self._heartbeat('WARMUP',bars=len(frame),reason='No validated instrument-aligned champion')
        if forecast.metadata.get('instrument_key')!=quote.symbol: return self._heartbeat('NO_ORDER',reason='Model instrument mismatch')
        decision_id=hashlib.sha1(f"{quote.symbol}:60s:{bar['event_time']}".encode()).hexdigest()[:20]
        with self.engine.store._lock,self.engine.store.conn:
            cur=self.engine.store.conn.execute("INSERT OR IGNORE INTO paper_decisions VALUES(?,?,?,'PROCESSING','{}')",(decision_id,quote.symbol,bar['event_time']))
            if cur.rowcount==0: return self._heartbeat('HEALTHY_NO_NEW_BAR')
        reference=(pd.Timestamp(bar['event_time'])+pd.Timedelta(seconds=60)).isoformat()
        target=(pd.Timestamp(reference)+pd.Timedelta(seconds=60)).isoformat()
        forecast_id=decision_id
        if hasattr(self.engine,'_persist_forecast'):
            self.engine._persist_forecast(forecast_id,now.isoformat(),target,'60s',bar['close'],forecast,{'pqrnn_intraday':1},'intraday',2*abs(forecast.p_up-.5),{'symbol':quote.symbol,'reference_time':reference,'experts':{'pqrnn_intraday':forecast.__dict__}})
        signal=self.bridge.signal(forecast,quote.symbol,quote.timestamp_ns,decision_id)
        snapshot=self.account.snapshot()
        current_notional=abs(position.quantity)*(quote.bid if position.quantity>=0 else quote.ask)
        other=max(0,snapshot['notional']-current_notional)
        order,decision=self.bridge.order(signal,quote,'order:'+decision_id,self.clock_ns(),current_notional,self.account.daily_pnl(quote.symbol),current_quantity=position.quantity,other_notional=other)
        self.engine.store.execute('UPDATE paper_decisions SET state=?,detail_json=? WHERE decision_id=?',('APPROVED' if order else 'REJECTED',json.dumps({'signal':signal.__dict__,'risk':decision.__dict__}),decision_id))
        if self._audit: self._audit.write('SIGNAL',signal=signal.__dict__,forecast=forecast.__dict__,risk=decision.__dict__)
        if order is None: return self._heartbeat('NO_ORDER',reason=decision.reason)
        return self._execute(order,decision_id,forecast_id)
