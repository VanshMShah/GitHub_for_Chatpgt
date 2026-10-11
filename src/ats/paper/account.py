"""Durable paper-only account. Orders are IOC; unfilled remainders are cancelled."""
from dataclasses import asdict
from datetime import datetime, timezone
from .types import Position, Side

class PaperAccount:
    def __init__(self, store, fee_bps=0.0, fixed_fee=0.0, starting_capital=100000.0):
        self.store, self.fee_bps, self.fixed_fee = store, float(fee_bps), float(fixed_fee)
        self.starting_capital = float(starting_capital)
        if min(self.fee_bps,self.fixed_fee,self.starting_capital) < 0:
            raise ValueError("Account capital and fee inputs must be nonnegative")

    def position(self, symbol):
        rows=self.store.query("SELECT * FROM paper_positions WHERE symbol=?",(symbol,))
        if not rows: return Position()
        r=rows[0]
        return Position(r['quantity'],r['average_price'],r['realized_pnl'])

    def begin_session(self,symbol,session):
        now=datetime.now(timezone.utc).isoformat()
        with self.store._lock,self.store.conn:
            self.store.conn.execute("INSERT OR IGNORE INTO paper_positions(symbol,quantity,average_price,realized_pnl,fees,session_date,session_start_pnl,updated_at) VALUES(?,0,0,0,0,?,0,?)",(symbol,session,now))
            r=self.store.conn.execute("SELECT * FROM paper_positions WHERE symbol=?",(symbol,)).fetchone()
            if r['session_date']!=session:
                unrealized=(r['mark']-r['average_price'])*r['quantity'] if r['mark'] is not None else 0
                self.store.conn.execute("UPDATE paper_positions SET session_date=?,session_start_pnl=?,updated_at=? WHERE symbol=?",(session,r['realized_pnl']-r['fees']+unrealized,now,symbol))

    def mark(self,symbol,price):
        self.store.execute("UPDATE paper_positions SET mark=?,updated_at=? WHERE symbol=?",(price,datetime.now(timezone.utc).isoformat(),symbol))

    def daily_pnl(self,symbol):
        r=self.store.query("SELECT * FROM paper_positions WHERE symbol=?",(symbol,))
        if not r: return 0.0
        r=r[0]
        unrealized=(r['mark']-r['average_price'])*r['quantity'] if r['mark'] is not None else 0
        return r['realized_pnl']-r['fees']+unrealized-r['session_start_pnl']

    def reserve(self,order,signal_id=None,forecast_id=None):
        with self.store._lock,self.store.conn:
            cur=self.store.conn.execute("INSERT OR IGNORE INTO paper_orders(order_id,signal_id,forecast_id,symbol,created_at,side,quantity,status) VALUES(?,?,?,?,?,?,?,'SUBMITTED')",(order.order_id,signal_id,forecast_id,order.symbol,datetime.now(timezone.utc).isoformat(),order.side.value,order.quantity))
            return cur.rowcount==1

    def reject(self,order,reason,signal_id=None,forecast_id=None):
        if self.reserve(order,signal_id,forecast_id): self.finish(order.order_id,'REJECTED',reason)

    def finish(self,order_id,status,reason):
        self.store.execute("UPDATE paper_orders SET status=?,reason=? WHERE order_id=? AND status='SUBMITTED'",(status,reason,order_id))

    def apply(self,order,fill):
        # Position, fee, fill and order status commit together; retries cannot double-apply.
        with self.store._lock,self.store.conn:
            c=self.store.conn
            if c.execute("SELECT 1 FROM paper_fills WHERE order_id=?",(order.order_id,)).fetchone():
                return self.position(order.symbol)
            r=c.execute("SELECT status FROM paper_orders WHERE order_id=?",(order.order_id,)).fetchone()
            if not r or r[0]!='SUBMITTED': raise ValueError('Order was not reserved for execution')
            fee=fill.quantity*fill.price*self.fee_bps/10000+self.fixed_fee
            position=self.position(order.symbol).apply(fill)
            now=datetime.now(timezone.utc).isoformat()
            c.execute("INSERT OR IGNORE INTO paper_positions(symbol,quantity,average_price,realized_pnl,fees,updated_at) VALUES(?,0,0,0,0,?)",(order.symbol,now))
            c.execute("UPDATE paper_positions SET quantity=?,average_price=?,realized_pnl=?,fees=fees+?,mark=?,updated_at=? WHERE symbol=?",(position.quantity,position.average_price,position.realized_pnl,fee,fill.price,now,order.symbol))
            c.execute("INSERT INTO paper_fills VALUES(?,?,?,?,?,?,?)",(order.order_id,fill.symbol,fill.timestamp_ns,fill.side.value,fill.quantity,fill.price,fee))
            status='FILLED' if fill.quantity==order.quantity else 'PARTIAL_CANCELLED'
            c.execute("UPDATE paper_orders SET status=?,reason=?,filled_quantity=?,fill_price=?,fee=? WHERE order_id=?",(status,'IOC remainder cancelled' if status=='PARTIAL_CANCELLED' else 'filled',fill.quantity,fill.price,fee,order.order_id))
            return position

    def recover(self):
        # A previous process may have stopped after reservation. Never resend its intent.
        from filelock import FileLock
        with FileLock(str(self.store.path)+".workflow.lock",timeout=600):
            self.store.execute("UPDATE paper_orders SET status='CANCELLED_ON_RESTART',reason='No committed fill; intent will not be resent' WHERE status='SUBMITTED'")

    def snapshot(self):
        positions=[]
        for row in self.store.query("SELECT * FROM paper_positions ORDER BY symbol"):
            r=dict(row)
            r['unrealized_pnl']=(r['mark']-r['average_price'])*r['quantity'] if r['mark'] is not None else None
            r['net_realized_pnl']=r['realized_pnl']-r['fees']
            r['notional']=abs(r['quantity'])*(r['mark'] or r['average_price'])
            positions.append(r)
        realized=sum(r['net_realized_pnl'] for r in positions)
        unrealized=sum(r['unrealized_pnl'] or 0 for r in positions)
        return {'mode':'PAPER_ONLY','starting_capital':self.starting_capital,'net_realized_pnl':realized,'unrealized_pnl':unrealized,
            'equity':self.starting_capital+realized+unrealized,'notional':sum(r['notional'] for r in positions),
            'fees':sum(r['fees'] for r in positions),'fee_bps':self.fee_bps,'fixed_fee':self.fixed_fee,
            'cost_note':'User-configured combined fee/tax estimate; zero means costs excluded.',
            'positions':positions,'orders':[dict(r) for r in self.store.query('SELECT * FROM paper_orders ORDER BY created_at DESC LIMIT 200')],
            'fills':[dict(r) for r in self.store.query('SELECT * FROM paper_fills ORDER BY timestamp_ns DESC LIMIT 200')]}
