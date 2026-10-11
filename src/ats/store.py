from __future__ import annotations
import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

class Store:
    """Single-machine durable store. Schema is portable to DuckDB later."""
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._local = threading.local()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._migrate()

    def _migrate(self) -> None:
        with self.conn:
            self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS observations (
              symbol TEXT NOT NULL, event_time TEXT NOT NULL, available_time TEXT NOT NULL,
              open REAL, high REAL, low REAL, close REAL NOT NULL, volume REAL,
              source TEXT NOT NULL, PRIMARY KEY(symbol, event_time)
            );
            CREATE TABLE IF NOT EXISTS intraday_observations (
              symbol TEXT NOT NULL, event_time TEXT NOT NULL, available_time TEXT NOT NULL,
              open REAL, high REAL, low REAL, close REAL NOT NULL, volume REAL, source TEXT NOT NULL,
              PRIMARY KEY(symbol, event_time)
            );
            CREATE TABLE IF NOT EXISTS live_quotes (
              symbol TEXT NOT NULL, event_time TEXT NOT NULL, available_time TEXT NOT NULL,
              price REAL NOT NULL, volume REAL, source TEXT NOT NULL,
              PRIMARY KEY(symbol, event_time)
            );
            CREATE TABLE IF NOT EXISTS exogenous (
              symbol TEXT NOT NULL, event_time TEXT NOT NULL, available_time TEXT NOT NULL,
              close REAL, source TEXT NOT NULL, PRIMARY KEY(symbol, event_time, source)
            );
            CREATE TABLE IF NOT EXISTS forecasts (
              forecast_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, target_time TEXT NOT NULL,
              horizon TEXT NOT NULL, model_version TEXT NOT NULL, policy_version TEXT NOT NULL,
              last_close REAL NOT NULL, mean REAL NOT NULL, p_up REAL NOT NULL,
              q05 REAL NOT NULL, q25 REAL NOT NULL, q50 REAL NOT NULL, q75 REAL NOT NULL, q95 REAL NOT NULL,
              volatility REAL NOT NULL, var95 REAL NOT NULL, es95 REAL NOT NULL,
              regime TEXT NOT NULL, conviction REAL NOT NULL, feature_json TEXT NOT NULL,
              settled INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS outcomes (
              forecast_id TEXT PRIMARY KEY, target_return REAL NOT NULL, actual_close REAL NOT NULL,
              settled_at TEXT NOT NULL, direction_correct INTEGER NOT NULL,
              abs_error REAL NOT NULL, squared_error REAL NOT NULL, brier REAL NOT NULL,
              interval_hit INTEGER NOT NULL, vol_proxy REAL
            );
            CREATE TABLE IF NOT EXISTS policy_state (
              key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS policy_history (
              id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, policy_version TEXT NOT NULL,
              context_json TEXT NOT NULL, weights_json TEXT NOT NULL, reason TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS diagnostics (
              id INTEGER PRIMARY KEY AUTOINCREMENT, forecast_id TEXT, timestamp TEXT NOT NULL,
              code TEXT NOT NULL, detail_json TEXT NOT NULL, action TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
              channel TEXT PRIMARY KEY, last_event_time TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
              event_id TEXT PRIMARY KEY, event_time TEXT NOT NULL, publication_time TEXT NOT NULL,
              available_time TEXT NOT NULL, source TEXT NOT NULL, category TEXT NOT NULL,
              severity REAL NOT NULL, novelty REAL NOT NULL, title TEXT, payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS instrument_bars (
              symbol TEXT NOT NULL, interval_seconds INTEGER NOT NULL,
              event_time TEXT NOT NULL, available_time TEXT NOT NULL,
              open REAL, high REAL, low REAL, close REAL NOT NULL, volume REAL, source TEXT,
              PRIMARY KEY(symbol,interval_seconds,event_time)
            );
            CREATE TABLE IF NOT EXISTS forecast_identity (
              forecast_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, reference_time TEXT NOT NULL,
              model_version TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS paper_positions (
              symbol TEXT PRIMARY KEY, quantity INTEGER NOT NULL, average_price REAL NOT NULL,
              realized_pnl REAL NOT NULL, fees REAL NOT NULL DEFAULT 0, session_date TEXT,
              session_start_pnl REAL NOT NULL DEFAULT 0, mark REAL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS paper_orders (
              order_id TEXT PRIMARY KEY, signal_id TEXT, forecast_id TEXT, symbol TEXT NOT NULL,
              created_at TEXT NOT NULL, side TEXT NOT NULL, quantity INTEGER NOT NULL,
              status TEXT NOT NULL, reason TEXT, filled_quantity INTEGER NOT NULL DEFAULT 0,
              fill_price REAL, fee REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS paper_fills (
              order_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, timestamp_ns INTEGER NOT NULL,
              side TEXT NOT NULL, quantity INTEGER NOT NULL, price REAL NOT NULL, fee REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS paper_decisions (
              decision_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, bar_time TEXT NOT NULL,
              state TEXT NOT NULL, detail_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_obs_symbol_time ON observations(symbol,event_time);
            CREATE INDEX IF NOT EXISTS idx_intraday_symbol_time ON intraday_observations(symbol,event_time);
            CREATE INDEX IF NOT EXISTS idx_live_symbol_time ON live_quotes(symbol,event_time);
            CREATE INDEX IF NOT EXISTS idx_fc_target ON forecasts(target_time,settled);
            CREATE INDEX IF NOT EXISTS idx_events_time ON events(event_time);
            ''')

    @contextmanager
    def transaction(self):
        with self._lock:
            depth=getattr(self._local,"depth",0)
            self._local.depth=depth+1
            try:
                if depth:
                    yield
                else:
                    with self.conn: yield
            finally:
                self._local.depth=depth

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self.transaction():
            self.conn.execute(sql,params)

    def executemany(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
        with self.transaction():
            self.conn.executemany(sql,rows)

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(sql, params).fetchall())

    def set_state(self, key: str, value: Any, now: str) -> None:
        self.execute("INSERT INTO policy_state(key,value_json,updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,updated_at=excluded.updated_at", (key,json.dumps(value, sort_keys=True),now))

    def get_state(self, key: str, default: Any = None) -> Any:
        rows = self.query("SELECT value_json FROM policy_state WHERE key=?", (key,))
        return default if not rows else json.loads(rows[0][0])

    def backup(self, destination: str | Path) -> str:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, sqlite3.connect(target) as backup:
            self.conn.backup(backup)
        return str(target)

    def close(self):
        with self._lock:
            self.conn.close()
