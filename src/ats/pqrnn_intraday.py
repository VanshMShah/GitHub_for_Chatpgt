from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ats.models import Forecast

try:
    import torch
    from torch import nn
    from src.ats.pqrnn import MicroPQRNN, PQRNNConfig, parameter_count, pinball
except Exception:  # pragma: no cover
    torch = None
    nn = None

UTC = timezone.utc


@dataclass(frozen=True)
class IntradayPQRNNConfig:
    feature_columns: tuple[str, ...] = ("ret", "ret3", "ret12", "rv12", "rv48", "range", "volume_z")
    sequence_length: int = 32
    horizon_bars: int = 1
    min_rows: int = 500
    holdout_rows: int = 100
    validation_fraction: float = 0.15
    train_epochs: int = 25
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 17
    min_bar_seconds: int = 55
    max_bar_seconds: int = 65
    model_name: str = "pqrnn_intraday_v1"


class IntradayPQRNNLifecycle:
    """Leakage-safe PQRNN lifecycle for bar-based intraday forecasts."""

    def __init__(self, engine, store, config: IntradayPQRNNConfig | None = None):
        self.engine = engine
        self.store = store
        self.cfg = config or IntradayPQRNNConfig()
        self.device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        from src.ats.config import settings
        model_dir = settings.model_dir
        model_dir.mkdir(parents=True, exist_ok=True)
        self.path = model_dir / "pqrnn_intraday_champion.pt"

    @property
    def available(self) -> bool:
        return torch is not None

    def _bar_seconds(self, frame: pd.DataFrame) -> float:
        ts = pd.to_datetime(frame["event_time"], utc=True).sort_values().drop_duplicates()
        if len(ts) < 3:
            raise RuntimeError("Need at least 3 timestamps to infer intraday resolution")
        gaps = ts.diff().dt.total_seconds().dropna().to_numpy()
        gaps = gaps[gaps > 0]
        if len(gaps) == 0:
            raise RuntimeError("Cannot infer intraday bar resolution")
        return float(np.median(gaps))

    def validate_resolution(self, frame: pd.DataFrame) -> float:
        seconds = self._bar_seconds(frame)
        if seconds < self.cfg.min_bar_seconds or seconds > self.cfg.max_bar_seconds:
            raise RuntimeError(
                f"Intraday PQRNN requires approximately 60-second bars; observed median resolution={seconds:.1f}s"
            )
        return seconds

    @staticmethod
    def make_supervised(frame: pd.DataFrame, horizon_bars: int) -> tuple[pd.DataFrame, tuple[str, ...]]:
        """Construct features at t and target from close at t+horizon.

        WHY: target construction is explicit and the target is removed from the feature frame.
        WATCH: overlapping horizons require a purge gap before validation/holdout evaluation.
        """
        x = frame.copy().sort_values("event_time").reset_index(drop=True)
        cols = ["ret", "ret3", "ret12", "rv12", "rv48", "range", "volume_z"]
        missing = [c for c in cols if c not in x.columns]
        if missing:
            raise RuntimeError(f"Missing intraday features: {missing}")
        x["target_end_time"] = pd.to_datetime(x["event_time"], utc=True).shift(-horizon_bars)
        future_close = x["close"].shift(-horizon_bars)
        x["target_ret"] = np.log(future_close / x["close"])
        x["target_up"] = (x["target_ret"] > 0).astype(float)
        expected_gap = pd.Timedelta(seconds=60*horizon_bars)
        x.loc[(x["target_end_time"]-pd.to_datetime(x["event_time"], utc=True)) != expected_gap, "target_ret"] = np.nan
        out = x.replace([np.inf,-np.inf],np.nan).dropna(subset=cols + ["target_ret", "target_end_time"]).copy().reset_index(drop=True)
        return out, tuple(cols)

    def _sequences(self, frame: pd.DataFrame):
        supervised, cols = self.make_supervised(frame, self.cfg.horizon_bars)
        if len(supervised) < self.cfg.min_rows:
            raise RuntimeError(f"Need {self.cfg.min_rows} usable intraday rows; have {len(supervised)}")

        seq = self.cfg.sequence_length
        values = supervised[list(cols)].to_numpy(dtype=np.float32)
        targets = supervised["target_ret"].to_numpy(dtype=np.float32)
        times = pd.to_datetime(supervised["event_time"], utc=True)
        ends = pd.to_datetime(supervised["target_end_time"], utc=True)
        indices = [i for i in range(seq-1,len(supervised)) if
                   (times.iloc[i-seq+1:i+1].diff().dropna().dt.total_seconds()==60).all()]
        if len(indices) < self.cfg.holdout_rows + 100:
            raise RuntimeError("Insufficient contiguous one-minute sequences")
        split = len(indices)-self.cfg.holdout_rows
        boundary = times.iloc[indices[split]]
        eligible = [j for j,i in enumerate(indices[:split]) if ends.iloc[i] < boundary]
        train_end = len(eligible)
        # holdout overlap purge removes only trailing samples.
        val_n = max(20,int(train_end*self.cfg.validation_fraction))
        train_n = train_end-val_n
        if train_n < 50:
            raise RuntimeError("Insufficient training rows after validation split")
        validation_boundary = times.iloc[indices[train_n]]
        optimizer_end = sum(ends.iloc[i] < validation_boundary for i in indices[:train_n])
        self.optimizer_end = optimizer_end
        self.validation_start = train_n
        scale_end = indices[optimizer_end-1]+1
        mu = values[:scale_end].mean(axis=0)
        sd = values[:scale_end].std(axis=0)
        sd = np.where(sd<1e-8,1.0,sd)
        xs = (values-mu)/sd
        X = np.asarray([xs[i-seq+1:i+1] for i in indices], dtype=np.float32)
        y = np.asarray([targets[i] for i in indices], dtype=np.float32)
        return X,y,train_end,split,mu,sd,cols

    def _model(self, features: int):
        return MicroPQRNN(PQRNNConfig(features=features)).to(self.device)

    def train(self, force: bool = False) -> dict[str, Any]:
        if not self.available:
            return {"status": "disabled", "reason": "PyTorch is unavailable"}
        if self.engine.is_market_hours():
            raise RuntimeError("Train intraday models outside market hours")
        raw = self.engine._load_intraday()
        instrument = raw.attrs.get("instrument_key")
        if not instrument:
            raise RuntimeError("Intraday training data must identify its instrument")
        frame = make_intraday_frame(raw)
        bar_seconds = self.validate_resolution(frame)
        X, y, train_end, holdout_start, mu, sd, cols = self._sequences(frame)

        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        model = self._model(X.shape[-1])
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        qlevels = model.cfg.quantiles
        loss_dir = nn.BCELoss()

        train_n = self.optimizer_end
        validation_start = self.validation_start
        return_scale = max(float(np.std(y[:train_n])),1e-5)
        y = y / return_scale
        if train_n < 50:
            raise RuntimeError("Insufficient training rows after validation split")
        tx = torch.from_numpy(X[:train_n]).to(self.device)
        ty = torch.from_numpy(y[:train_n]).to(self.device)
        vx = torch.from_numpy(X[validation_start:train_end]).to(self.device)
        vy = torch.from_numpy(y[validation_start:train_end]).to(self.device)
        best_loss = math.inf
        best_state = None
        best_epoch = -1
        for epoch in range(self.cfg.train_epochs):
            perm = torch.randperm(len(tx), device=self.device)
            model.train()
            for start in range(0, len(tx), self.cfg.batch_size):
                idx = perm[start:start + self.cfg.batch_size]
                out = model(tx[idx])
                q = out["quantiles"]
                loss = (
                    0.60 * pinball(q, ty[idx], qlevels)
                    + 0.30 * loss_dir(out["p_up"], (ty[idx] > 0).float())
                    + 0.10 * torch.mean((torch.log1p(out["volatility"]) - torch.log1p(ty[idx].abs())).pow(2))
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                out = model(vx)
                val_loss = float((
                    0.60 * pinball(out["quantiles"], vy, qlevels)
                    + 0.30 * loss_dir(out["p_up"], (vy > 0).float())
                    + 0.10 * torch.mean((torch.log1p(out["volatility"]) - torch.log1p(vy.abs())).pow(2))
                ).cpu())
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        assert best_state is not None
        model.load_state_dict(best_state)
        model.eval()
        with torch.inference_mode():
            out = model(torch.from_numpy(X[holdout_start:]).to(self.device))
            p = out["p_up"].detach().cpu().numpy()
            q = out["quantiles"].detach().cpu().numpy()
            vol = out["volatility"].detach().cpu().numpy()
        q = q * return_scale
        vol = vol * return_scale
        y_hold = y[holdout_start:] * return_scale
        metrics = {
            "directional_accuracy": float(np.mean((p >= 0.5) == (y_hold > 0))),
            "mae": float(np.mean(np.abs(y_hold - q[:, 2]))),
            "brier": float(np.mean((p - (y_hold > 0)) ** 2)),
            "p90_coverage": float(np.mean((y_hold >= q[:, 0]) & (y_hold <= q[:, -1]))),
            "vol_mae": float(np.mean(np.abs(np.abs(y_hold) - vol))),
        }
        base_probability = float(np.clip(np.mean(y[:train_n]>0),0.01,0.99))
        baseline_brier = float(np.mean((base_probability-(y_hold>0))**2))
        baseline_mae = float(np.mean(np.abs(y_hold)))
        promoted = (metrics["brier"] < baseline_brier-0.005 and metrics["mae"] <= baseline_mae*1.01
                    and 0.80 <= metrics["p90_coverage"] <= 0.99)
        meta = {
            "schema_version":2,
            "instrument_key":instrument,
            "interval_seconds":60,
            "return_scale":return_scale,
            "promoted":bool(promoted),
            "baseline":{"brier":baseline_brier,"mae":baseline_mae},
            "evaluation_note":"fixed chronological holdout; require an independent later paper evaluation",
            "model_name": self.cfg.model_name,
            "bar_seconds": bar_seconds,
            "horizon_bars": self.cfg.horizon_bars,
            "horizon_seconds": int(round(bar_seconds * self.cfg.horizon_bars)),
            "sequence_length": self.cfg.sequence_length,
            "features": list(cols),
            "parameter_count": parameter_count(model),
            "best_epoch": best_epoch,
            "validation_loss": best_loss,
            "holdout_rows": len(y_hold),
            "metrics": metrics,
            "trained_at": datetime.now(UTC).isoformat(),
            "status": "promoted" if promoted else "rejected",
        }
        artifact = self.path if promoted else self.path.with_name("pqrnn_intraday_challenger.pt")
        temporary = artifact.with_suffix(".tmp")
        torch.save({"state_dict": model.state_dict(), "config": model.cfg.__dict__, "feature_columns": list(cols), "mean": mu.tolist(), "std": sd.tolist(), "meta": meta}, temporary)
        temporary.replace(artifact)
        self.store.set_state("pqrnn_intraday_last_run", meta, meta["trained_at"])
        if promoted:
            self.store.set_state("pqrnn_intraday_meta", meta, meta["trained_at"])
        return meta

    def predict(self, frame: pd.DataFrame) -> Forecast | None:
        if not self.available or not self.path.exists():
            return None
        checkpoint = torch.load(self.path, map_location=self.device, weights_only=False)
        meta = checkpoint["meta"]
        if meta.get("schema_version") != 2 or not meta.get("promoted"):
            raise RuntimeError("Legacy or unvalidated intraday checkpoint: train and pass validation gates")
        if frame.attrs.get("instrument_key") != meta.get("instrument_key"):
            raise RuntimeError("Intraday model/execution instrument mismatch")
        bar_seconds = self.validate_resolution(frame)
        if int(round(bar_seconds * meta["horizon_bars"])) > 60:
            raise RuntimeError("Intraday PQRNN horizon exceeds the 60-second paper-trading boundary")
        # Prediction uses features from the latest known row only. Target columns
        # are deliberately ignored and may be NaN at live time.
        raw = make_intraday_frame(frame).sort_values("event_time").reset_index(drop=True)
        if len(raw) < int(meta["sequence_length"]):
            return None
        recent_times = pd.to_datetime(raw["event_time"].iloc[-int(meta["sequence_length"]):], utc=True)
        if not (recent_times.diff().dropna().dt.total_seconds()==60).all():
            raise RuntimeError("Intraday inference requires contiguous one-minute context")
        feature_cols = list(checkpoint["feature_columns"])
        x = raw[feature_cols].to_numpy(dtype=np.float32)
        mu = np.asarray(checkpoint["mean"], dtype=np.float32)
        sd = np.asarray(checkpoint["std"], dtype=np.float32)
        x = (x - mu) / np.where(sd < 1e-8, 1.0, sd)
        if not np.isfinite(x[-int(meta["sequence_length"]):]).all():
            raise RuntimeError("Intraday features contain missing or nonfinite values")
        model = self._model(len(feature_cols))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.inference_mode():
            out = model(torch.from_numpy(x[-int(meta["sequence_length"]):][None, ...]).to(self.device))
        q = out["quantiles"][0].detach().cpu().numpy() * meta["return_scale"]
        p = float(out["p_up"][0].detach().cpu())
        vol = float(out["volatility"][0].detach().cpu()) * meta["return_scale"]
        horizon = f"{int(round(bar_seconds * meta['horizon_bars']))}s"
        return Forecast(float(q[2]), p, float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4]), max(vol, 1e-8), {"expert": "pqrnn_intraday", "horizon": horizon, "model": meta["model_name"],"instrument_key":meta["instrument_key"],"point_estimate":"median"})


def make_intraday_frame(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy().sort_values("event_time").reset_index(drop=True)
    if x.empty:
        return x
    x["event_time"] = pd.to_datetime(x["event_time"], utc=True)
    groups=x["event_time"].diff().dt.total_seconds().ne(60).cumsum()
    close=x["close"].groupby(groups)
    x["ret"] = np.log(x["close"] / close.shift(1))
    x["ret3"] = np.log(x["close"] / close.shift(3))
    x["ret12"] = np.log(x["close"] / close.shift(12))
    returns=x["ret"].groupby(groups)
    x["rv12"] = returns.transform(lambda s:s.rolling(12).std())
    x["rv48"] = returns.transform(lambda s:s.rolling(48).std())
    x["range"] = np.log(x["high"] / x["low"])
    volumes=x["volume"].groupby(groups)
    mean=volumes.transform(lambda s:s.rolling(48).mean())
    std=volumes.transform(lambda s:s.rolling(48).std())
    x["volume_z"] = (x["volume"]-mean)/(std+1e-12)
    return x
