from __future__ import annotations
import numpy as np
import pandas as pd

SYMBOLS = {
    "nifty": "^NSEI", "india_vix": "^INDIAVIX", "sp500": "^GSPC", "vix": "^VIX",
    "brent": "BZ=F", "wti": "CL=F", "gold": "GC=F", "copper": "HG=F", "usdinr": "INR=X", "dxy": "DX-Y.NYB",
}

def logret(s: pd.Series, n: int = 1) -> pd.Series:
    return np.log(s / s.shift(n))

def make_daily_features(nifty: pd.DataFrame, exo: dict[str,pd.DataFrame]) -> pd.DataFrame:
    n = nifty.copy().set_index("event_time").sort_index()
    daily = n.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna(subset=["close"])
    daily["ret1"] = logret(daily["close"])
    daily["ret5"] = logret(daily["close"],5)
    daily["ret22"] = logret(daily["close"],22)
    daily["range"] = np.log(daily["high"] / daily["low"])
    daily["rv_5"] = daily["ret1"].rolling(5).std()
    daily["rv_22"] = daily["ret1"].rolling(22).std()
    daily["momentum_5"] = daily["ret1"].rolling(5).mean()
    daily["momentum_20"] = daily["ret1"].rolling(20).mean()
    daily["drawdown_20"] = daily["close"] / daily["close"].rolling(20).max() - 1.0
    for name, frame in exo.items():
        if frame.empty:
            continue
        x = frame.copy().set_index("event_time").sort_index()["close"].resample("1D").last().ffill()
        daily[name] = logret(x).shift(1).reindex(daily.index).ffill()
        daily[f"{name}_shock"] = daily[name].rolling(5).std()
    if "india_vix" in exo and not exo["india_vix"].empty:
        v = exo["india_vix"].copy().set_index("event_time")["close"].resample("1D").last().ffill()
        daily["vix_level"] = v.reindex(daily.index).ffill()
        daily["vix_change"] = logret(v).reindex(daily.index).ffill()
    daily["target_ret"] = daily["ret1"].shift(-1)
    daily["target_up"] = (daily["target_ret"] > 0).astype(float)
    return daily.reset_index()

FEATURES = [
    "ret1","ret5","ret22","range","rv_5","rv_22","momentum_5","momentum_20","drawdown_20",
    "india_vix","india_vix_shock","vix_level","vix_change","brent","brent_shock","wti","wti_shock",
    "gold","gold_shock","copper","copper_shock","usdinr","usdinr_shock","dxy","dxy_shock","sp500","sp500_shock","vix","vix_shock"
]

def clean_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FEATURES if c in frame.columns]
    out = frame.dropna(subset=cols + ["target_ret"]).copy()
    return out

def make_intraday_features(frame: pd.DataFrame, horizon_bars: int = 2) -> pd.DataFrame:
    x = frame.copy().sort_values('event_time').set_index('event_time')
    x['ret'] = logret(x['close'])
    x['ret3'] = logret(x['close'], 3)
    x['ret12'] = logret(x['close'], 12)
    x['rv12'] = x['ret'].rolling(12).std()
    x['rv48'] = x['ret'].rolling(48).std()
    x['range'] = np.log(x['high'] / x['low'])
    x['volume_z'] = (x['volume'] - x['volume'].rolling(48).mean()) / (x['volume'].rolling(48).std() + 1e-12)
    future = x['close'].shift(-horizon_bars)
    x['target_ret'] = np.log(future / x['close'])
    x['target_up'] = (x['target_ret'] > 0).astype(float)
    return x.reset_index()
