from __future__ import annotations

import json
import math
import os
import secrets
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.ats.features import FEATURES, clean_training_frame, make_daily_features
from src.ats.models import Forecast

try:
    import torch
    from torch import nn
    from src.ats.pqrnn import MicroPQRNN, PQRNNConfig, parameter_count, pinball
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None

UTC = timezone.utc


@dataclass(frozen=True)
class PQRNNLifecycleConfig:
    min_outcomes: int = 80
    min_train_rows: int = 320
    sequence_length: int = 20
    holdout_rows: int = 60
    train_epochs: int = 35
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    trigger_loss_ratio: float = 1.15
    trigger_recent_window: int = 30
    retrain_cooldown_outcomes: int = 30
    challenger_margin: float = 0.01
    bootstrap_reps: int = 300
    block_length: int = 5
    seed: int = 7


class PQRNNLifecycle:
    """Champion/challenger lifecycle for the optional Micro-PQRNN.

    Promotion is deliberately conservative:
    1. evidence trigger fires
    2. train candidate on a point-in-time daily sequence
    3. evaluate candidate and incumbent on the same holdout
    4. require paired bootstrap evidence plus practical margin
    5. promote only when the challenger is materially better
    """

    def __init__(self, engine, store, config: PQRNNLifecycleConfig | None = None):
        self.engine = engine
        self.store = store
        self.cfg = config or PQRNNLifecycleConfig()
        self.model_dir = Path(os.getenv("ATS_MODEL_DIR", "models"))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.champion_path = self.model_dir / "pqrnn_champion.pt"
        self.challenger_path = self.model_dir / "pqrnn_challenger.pt"
        self.meta_path = self.model_dir / "pqrnn_meta.json"
        self.device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        self._run_lock = threading.Lock()

    @property
    def available(self) -> bool:
        return torch is not None

    def _save_meta(self, meta: dict[str, Any]) -> None:
        self.meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        self.store.set_state("pqrnn_meta", meta, datetime.now(UTC).isoformat())

    def _latest_outcome_count(self) -> int:
        row = self.store.query("SELECT COUNT(*) FROM outcomes o JOIN forecasts f USING(forecast_id) WHERE f.horizon='1D'")[0]
        return int(row[0])

    def _recent_error_pressure(self) -> float:
        n = max(10, self.cfg.trigger_recent_window)
        rows = self.store.query(
            "SELECT o.abs_error FROM outcomes o JOIN forecasts f USING(forecast_id) WHERE f.horizon='1D' ORDER BY o.settled_at DESC LIMIT ?", (n * 2,)
        )
        vals = np.asarray([float(r[0]) for r in rows], dtype=float)
        if len(vals) < n:
            return 1.0
        recent = float(np.mean(vals[:n]))
        prior = float(np.mean(vals[n:]))
        return recent / max(prior, 1e-9)

    def trigger_status(self) -> dict[str, Any]:
        outcomes = self._latest_outcome_count()
        meta = self.store.get_state("pqrnn_meta", {})
        since = int(meta.get("last_train_outcomes", 0) or 0)
        new_outcomes = max(0, outcomes - since)
        pressure = self._recent_error_pressure()
        diagnostics = self.store.query(
            "SELECT COUNT(*) FROM diagnostics d JOIN forecasts f USING(forecast_id) WHERE f.horizon='1D' AND d.code IN ('SYSTEMATIC_DRIFT','DISTRIBUTION_MISS','MAGNITUDE_MISS')"
        )[0][0]
        enough = outcomes >= self.cfg.min_outcomes
        cooldown_ok = new_outcomes >= self.cfg.retrain_cooldown_outcomes
        drift = pressure >= self.cfg.trigger_loss_ratio
        diagnostic_pressure = int(diagnostics) >= 5
        trigger = enough and cooldown_ok and (drift or diagnostic_pressure)
        return {
            "available": self.available,
            "device": self.device,
            "outcomes": outcomes,
            "new_outcomes_since_training": new_outcomes,
            "recent_error_ratio": pressure,
            "diagnostic_pressure": int(diagnostics),
            "trigger": trigger,
            "reason": "systematic_loss_pressure" if drift else ("diagnostic_pressure" if diagnostic_pressure else "none"),
        }

    def _frame(self):
        n, e = self.engine._load()
        f = clean_training_frame(make_daily_features(n, e)).reset_index(drop=True)
        cols = [c for c in FEATURES if c in f.columns]
        return f, cols

    def _sequence_data(self, f, cols):
        seq = self.cfg.sequence_length
        X = f[cols].to_numpy(dtype=np.float32)
        y = f["target_ret"].to_numpy(dtype=np.float32)
        if len(X) < self.cfg.min_train_rows or len(X) <= seq + self.cfg.holdout_rows:
            raise RuntimeError(f"Need at least {self.cfg.min_train_rows} usable rows for PQRNN; have {len(X)}")
        # Standardize with train-only statistics to avoid leakage.
        cutoff = len(X) - self.cfg.holdout_rows
        split = cutoff - seq + 1
        val_n = max(10, int(split * 0.15))
        train_n = split - val_n
        if train_n < 30:
            train_n = split
        scale_end = seq - 1 + train_n
        mu = X[:scale_end].mean(axis=0)
        sd = X[:scale_end].std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xs = (X - mu) / sd
        xs, ys = [], []
        for i in range(seq-1, len(Xs)):
            xs.append(Xs[i-seq+1:i+1])
            ys.append(y[i])
        split = cutoff - seq + 1
        return np.asarray(xs), np.asarray(ys), split, mu.astype(np.float32), sd.astype(np.float32)

    def _make_model(self, features: int):
        cfg = PQRNNConfig(features=features)
        return MicroPQRNN(cfg).to(self.device)

    def _metrics(self, p: np.ndarray, q: np.ndarray, vol: np.ndarray, y: np.ndarray) -> dict[str, float]:
        p = np.clip(p, 1e-5, 1 - 1e-5)
        direction = (p >= 0.5).astype(int)
        actual = (y > 0).astype(int)
        brier = float(np.mean((p - actual) ** 2))
        ll = float(-np.mean(actual * np.log(p) + (1 - actual) * np.log(1 - p)))
        mae = float(np.mean(np.abs(y - q[:, 2])))
        hit = float(np.mean((y >= q[:, 0]) & (y <= q[:, -1])))
        return {
            "directional_accuracy": float(np.mean(direction == actual)),
            "brier": brier,
            "log_loss": ll,
            "mae": mae,
            "p90_coverage": hit,
            "vol_mae": float(np.mean(np.abs(np.abs(y) - vol))),
        }

    def _bootstrap_pvalue(self, candidate_loss: np.ndarray, incumbent_loss: np.ndarray) -> tuple[float, float, float]:
        d = np.asarray(candidate_loss) - np.asarray(incumbent_loss)
        if len(d) < 20:
            return 1.0, float(np.mean(d)), float(np.mean(d))
        rng = np.random.default_rng(self.cfg.seed)
        n = len(d)
        block = max(1, min(self.cfg.block_length, n))
        n_blocks = math.ceil(n / block)
        samples = np.empty(self.cfg.bootstrap_reps, dtype=float)
        for _ in range(self.cfg.bootstrap_reps):
            starts = rng.integers(0, n - block + 1, size=n_blocks)
            idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
            samples[_] = d[idx].mean()
        p = float(np.mean(samples >= 0.0))
        lo, hi = np.quantile(samples, [0.05, 0.95])
        return p, float(lo), float(hi)

    def _bootstrap_pvalue_iid_reference(
        self, candidate_loss: np.ndarray, incumbent_loss: np.ndarray
    ) -> tuple[float, float, float]:
        d = np.asarray(candidate_loss) - np.asarray(incumbent_loss)
        if len(d) < 20:
            return 1.0, float(np.mean(d)), float(np.mean(d))
        rng = np.random.default_rng(self.cfg.seed)
        samples = rng.choice(d, size=(self.cfg.bootstrap_reps, len(d)), replace=True).mean(axis=1)
        p = float(np.mean(samples >= 0.0))
        lo, hi = np.quantile(samples, [0.05, 0.95])
        return p, float(lo), float(hi)

    def _train_candidate(self, x: np.ndarray, y: np.ndarray, split: int):
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        model = self._make_model(x.shape[-1])
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        qlevels = model.cfg.quantiles
        loss_dir = nn.BCELoss()
        val_frac = 0.15
        val_n = max(10, int(split * val_frac))
        train_n = split - val_n
        if train_n < 30:
            val_n = 0
            train_n = split

        tx = torch.from_numpy(x[:train_n]).to(self.device)
        ty = torch.from_numpy(y[:train_n]).to(self.device)
        vx = torch.from_numpy(x[train_n:split]).to(self.device) if val_n else None
        vy = torch.from_numpy(y[train_n:split]).to(self.device) if val_n else None
        best_state = None
        best = math.inf
        best_epoch = -1
        n = len(tx)
        for epoch in range(self.cfg.train_epochs):
            perm = torch.randperm(n, device=self.device)
            model.train()
            for start in range(0, n, self.cfg.batch_size):
                idx = perm[start:start+self.cfg.batch_size]
                out = model(tx[idx])
                q = out["quantiles"]
                qloss = pinball(q, ty[idx], qlevels)
                dloss = loss_dir(out["p_up"], (ty[idx] > 0).float())
                vloss = torch.mean((torch.log1p(out["volatility"]) - torch.log1p(ty[idx].abs())).pow(2))
                loss = 0.60 * qloss + 0.30 * dloss + 0.10 * vloss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            with torch.no_grad():
                if val_n:
                    out = model(vx)
                    q = out["quantiles"]
                    qloss = pinball(q, vy, qlevels)
                    dloss = loss_dir(out["p_up"], (vy > 0).float())
                    vloss = torch.mean((torch.log1p(out["volatility"]) - torch.log1p(vy.abs())).pow(2))
                    check_loss = float((0.60 * qloss + 0.30 * dloss + 0.10 * vloss).cpu())
                else:
                    out = model(tx)
                    q = out["quantiles"]
                    qloss = pinball(q, ty, qlevels)
                    dloss = loss_dir(out["p_up"], (ty > 0).float())
                    vloss = torch.mean((torch.log1p(out["volatility"]) - torch.log1p(ty.abs())).pow(2))
                    check_loss = float((0.60 * qloss + 0.30 * dloss + 0.10 * vloss).cpu())

            if check_loss < best:
                best = check_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        self._last_checkpoint_epoch = best_epoch
        self._last_checkpoint_used_validation = bool(val_n)
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model, best

    def _incumbent_holdout(self, f, cols, split):
        # Refit incumbent experts only on data available before each holdout point,
        # then score the entire holdout one point at a time. This avoids hindsight.
        from src.ats.models import PersistenceExpert, LogisticExpert, QuantileGBDT, ARIMAGARCHExpert, blend
        from src.ats.policy import DMAPolicy

        policy = DMAPolicy()
        start = self.cfg.sequence_length - 1 + split
        preds, actual = [], []
        for i in range(start, len(f)):
            train = f.iloc[:i]
            test = f.iloc[i:i+1]
            X = train[cols].to_numpy(); y = train["target_ret"].to_numpy(); Xt = test[cols].to_numpy()
            experts = {
                "persistence": PersistenceExpert().fit(y).predict(last_return=float(train["ret1"].iloc[-1])),
                "logistic": LogisticExpert().fit(X, y).predict(Xt),
                "lightgbm_compatible": QuantileGBDT().fit(X, y).predict(Xt),
                "arima_gjr_garch": ARIMAGARCHExpert().fit(y[-756:]).predict(),
            }
            ctx = {"vix": float(test.get("vix_level", 0).iloc[0]) if "vix_level" in test else 0.0,
                   "rv22": float(test.get("rv_22", 0).iloc[0]) if "rv_22" in test else 0.0,
                   "momentum20": float(test.get("momentum_20", 0).iloc[0]) if "momentum_20" in test else 0.0}
            w = policy.choose(experts, ctx)
            out = blend(experts, w)
            preds.append(out)
            actual.append(float(test["target_ret"].iloc[0]))
            policy.observe(actual[-1], experts)
        return preds, np.asarray(actual, dtype=float)

    def train_and_evaluate(self, force: bool = False) -> dict[str, Any]:
        if self.engine.is_market_hours():
            raise RuntimeError("Train daily neural challengers outside market hours")
        if not self.available:
            return {"status": "disabled", "reason": "PyTorch optional dependency is not installed."}
        if not self._run_lock.acquire(blocking=False):
            return {"status": "busy", "reason": "PQRNN lifecycle already running"}
        try:
            return self._train_and_evaluate_locked(force)
        finally:
            self._run_lock.release()

    def _train_and_evaluate_locked(self, force: bool = False) -> dict[str, Any]:
        trigger = self.trigger_status()
        if not force and not trigger["trigger"]:
            return {"status": "not_triggered", **trigger}

        f, cols = self._frame()
        x, y, split, mu, sd = self._sequence_data(f, cols)
        model, train_loss = self._train_candidate(x, y, split)
        with torch.no_grad():
            tx = torch.from_numpy(x[split:]).to(self.device)
            out = model(tx)
            p = out["p_up"].detach().cpu().numpy()
            q = out["quantiles"].detach().cpu().numpy()
            vol = out["volatility"].detach().cpu().numpy()
        y_hold = y[split:]
        candidate = self._metrics(p, q, vol, y_hold)

        incumbent_fc, incumbent_y = self._incumbent_holdout(f, cols, split)
        ip = np.asarray([v.p_up for v in incumbent_fc])
        iq = np.asarray([[v.q05, v.q25, v.q50, v.q75, v.q95] for v in incumbent_fc])
        iv = np.asarray([v.volatility for v in incumbent_fc])
        incumbent = self._metrics(ip, iq, iv, incumbent_y)
        n = min(len(p), len(ip), len(y_hold), len(incumbent_y))
        p, ip, y_al = p[-n:], ip[-n:], y_hold[-n:]
        cand_logloss = -( (y_al > 0) * np.log(np.clip(p,1e-5,1-1e-5)) + (y_al <= 0) * np.log(np.clip(1-p,1e-5,1-1e-5)) )
        inc_logloss = -( (y_al > 0) * np.log(np.clip(ip,1e-5,1-1e-5)) + (y_al <= 0) * np.log(np.clip(1-ip,1e-5,1-1e-5)) )
        pval, ci_lo, ci_hi = self._bootstrap_pvalue(cand_logloss, inc_logloss)
        mean_delta = float(np.mean(cand_logloss - inc_logloss))
        practical = mean_delta <= -self.cfg.challenger_margin
        statistically = ci_hi < 0.0 and pval < 0.10
        promote = practical and statistically and candidate["mae"] <= incumbent["mae"] * 1.01

        stamp = datetime.now(UTC).isoformat()
        run_id = secrets.token_hex(8)
        state = {
            "run_id": run_id,
            "schema_version":2,
            "evaluation_note":"chronological holdout; repeated promotion attempts reuse holdout and require a separate final evaluation",
            "timestamp": stamp,
            "device": self.device,
            "features": cols,
            "sequence_length": self.cfg.sequence_length,
            "parameter_count": parameter_count(model),
            "train_loss": train_loss,
            "candidate": candidate,
            "incumbent": incumbent,
            "paired_logloss_delta": mean_delta,
            "bootstrap_pvalue": pval,
            "bootstrap_ci90": [ci_lo, ci_hi],
            "practical_margin": self.cfg.challenger_margin,
            "promote": promote,
            "status": "promoted" if promote else "rejected",
        }

        torch.save({"state_dict": model.state_dict(), "config": asdict(model.cfg), "feature_columns": cols,
                    "mean": mu.tolist(), "std": sd.tolist(), "meta": state}, self.challenger_path)
        self._save_meta({**state, "last_train_outcomes": self._latest_outcome_count()})
        self.store.set_state("pqrnn_last_run", state, stamp)

        if promote:
            os.replace(self.challenger_path, self.champion_path)
            promoted = {**state, "artifact": str(self.champion_path)}
            self._save_meta({**promoted, "champion": True, "last_train_outcomes": self._latest_outcome_count()})
            self.store.set_state("pqrnn_champion", promoted, stamp)
            self.store.execute(
                "INSERT INTO diagnostics(forecast_id,timestamp,code,detail_json,action) VALUES(?,?,?,?,?)",
                (None, stamp, "PQRNN_PROMOTED", json.dumps(state), "Promoted challenger after out-of-sample statistical gate"),
            )
        else:
            self.store.execute(
                "INSERT INTO diagnostics(forecast_id,timestamp,code,detail_json,action) VALUES(?,?,?,?,?)",
                (None, stamp, "PQRNN_REJECTED", json.dumps(state), "Rejected challenger; incumbent retained"),
            )
            self.store.set_state("pqrnn_rejection", state, stamp)
        return state


    def predict(self, frame, cols):
        """Return the promoted PQRNN forecast, or None when no champion exists."""
        if not self.available or not self.champion_path.exists():
            return None
        try:
            checkpoint = torch.load(self.champion_path, map_location=self.device, weights_only=False)
            if checkpoint.get("meta", {}).get("schema_version") != 2:
                return None
            features = list(checkpoint.get("feature_columns", cols))
            if any(c not in frame.columns for c in features) or len(frame) < self.cfg.sequence_length:
                return None
            model_cfg = PQRNNConfig(**checkpoint["config"])
            model = MicroPQRNN(model_cfg).to(self.device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            x = frame[features].to_numpy(dtype=np.float32)
            mu = np.asarray(checkpoint["mean"], dtype=np.float32)
            sd = np.asarray(checkpoint["std"], dtype=np.float32)
            xs = (x - mu) / np.where(sd < 1e-8, 1.0, sd)
            tx = torch.from_numpy(xs[-self.cfg.sequence_length:][None, ...]).to(self.device)
            with torch.inference_mode():
                out = model(tx)
            q = out["quantiles"][0].detach().cpu().numpy()
            p = float(out["p_up"][0].detach().cpu())
            vol = float(out["volatility"][0].detach().cpu())
            return Forecast(float(q[2]), p, float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4]), max(vol, 1e-6), {
                "expert": "pqrnn", "artifact": str(self.champion_path), "champion_run_id": self.store.get_state("pqrnn_champion", {}).get("run_id"),
                "horizon": "1D"
            })
        except Exception as exc:
            stamp = datetime.now(UTC).isoformat()
            self.store.set_state("pqrnn_inference_error", {"error": str(exc), "updated_at": stamp}, stamp)
            self.store.execute(
                "INSERT INTO diagnostics(forecast_id,timestamp,code,detail_json,action) VALUES(?,?,?,?,?)",
                (None, stamp, "PQRNN_INFERENCE_FAILED", json.dumps({"error": str(exc)}), "Excluded PQRNN; incumbent ensemble retained"),
            )
            return None

    def status(self) -> dict[str, Any]:
        s = self.trigger_status()
        s.update({
            "champion_exists": self.champion_path.exists(),
            "last_run": self.store.get_state("pqrnn_last_run"),
            "champion": self.store.get_state("pqrnn_champion"),
            "eligible_in_ensemble": self.champion_path.exists(),
            "last_inference_error": self.store.get_state("pqrnn_inference_error"),
        })
        return s
