from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error

from src.ats.data import YahooChartProvider, bars_to_frame
from src.ats.features import SYMBOLS, FEATURES, make_daily_features, clean_training_frame
from src.ats.models import PersistenceExpert, LogisticExpert, QuantileGBDT, ARIMAGARCHExpert, blend
from src.ats.policy import DMAPolicy


def _loss(y, f):
    return float(mean_absolute_error(y, [x.mean for x in f]))


def evaluate(history_years: str = "5y", min_train: int = 252, max_cases: int = 300) -> dict:
    provider = YahooChartProvider()
    nifty = bars_to_frame(provider.fetch(SYMBOLS["nifty"], period=history_years, interval="1d"))
    exo = {}
    for name, symbol in SYMBOLS.items():
        if name == "nifty":
            continue
        try:
            exo[name] = bars_to_frame(provider.fetch(symbol, period=history_years, interval="1d"))[["event_time", "close"]]
        except Exception:
            exo[name] = bars_to_frame([])

    frame = clean_training_frame(make_daily_features(nifty, exo)).reset_index(drop=True)
    cols = [c for c in FEATURES if c in frame.columns]
    if len(frame) < min_train + 5:
        raise RuntimeError(f"Need at least {min_train + 5} usable rows; have {len(frame)}")

    start = max(min_train, len(frame) - max_cases)
    actual_r: list[float] = []
    actual: list[int] = []
    probs = {k: [] for k in ("persistence", "logistic", "lightgbm_compatible", "arima_gjr_garch", "dma")}
    maes = {k: [] for k in probs}
    policy = DMAPolicy()
    policy_paths = []

    for i in range(start, len(frame)):
        train = frame.iloc[:i]
        test = frame.iloc[i:i + 1]
        X = train[cols].to_numpy()
        y = train["target_ret"].to_numpy()
        Xt = test[cols].to_numpy()
        experts = {
            "persistence": PersistenceExpert().fit(y),
            "logistic": LogisticExpert().fit(X, y),
            "lightgbm_compatible": QuantileGBDT().fit(X, y),
            "arima_gjr_garch": ARIMAGARCHExpert().fit(y[-min(len(y), 756):]),
        }
        fs = {
            "persistence": experts["persistence"].predict(last_return=float(train["ret1"].iloc[-1])),
            "logistic": experts["logistic"].predict(Xt),
            "lightgbm_compatible": experts["lightgbm_compatible"].predict(Xt),
            "arima_gjr_garch": experts["arima_gjr_garch"].predict(),
        }
        ctx = {
            "vix": float(test["vix_level"].iloc[0]) if "vix_level" in test else 0.0,
            "rv22": float(test["rv_22"].iloc[0]) if "rv_22" in test else 0.0,
            "momentum20": float(test["momentum_20"].iloc[0]) if "momentum_20" in test else 0.0,
        }
        w = policy.choose(fs, ctx)
        dma = blend(fs, w)
        ar = float(test["target_ret"].iloc[0])
        actual_r.append(ar)
        actual.append(int(ar > 0))
        for k, f in fs.items():
            probs[k].append(float(f.p_up))
            maes[k].append(abs(ar - f.mean))
        probs["dma"].append(float(dma.p_up))
        maes["dma"].append(abs(ar - dma.mean))
        policy.observe(ar, fs, reason="walk-forward outcome")
        policy_paths.append(dict(policy.weights))

    a = np.asarray(actual)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n": len(actual),
        "history_years": history_years,
        "min_train": min_train,
        "cases": max_cases,
        "models": {},
        "dma_final_weights": policy.weights,
        "dma_weight_path_tail": policy_paths[-10:],
    }
    for name, ps in probs.items():
        p = np.clip(np.asarray(ps), 1e-6, 1 - 1e-6)
        result["models"][name] = {
            "directional_accuracy": float(accuracy_score(a, p >= 0.5)),
            "brier": float(brier_score_loss(a, p)),
            "log_loss": float(log_loss(a, p, labels=[0, 1])),
            "mean_absolute_error": float(np.mean(maes[name])),
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="ATS strict walk-forward NIFTY expert + DMA benchmark")
    ap.add_argument("--years", default="5y")
    ap.add_argument("--min-train", type=int, default=252)
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--out", default="research/walk_forward.json")
    args = ap.parse_args()
    result = evaluate(args.years, args.min_train, args.cases)
    out = __import__("pathlib").Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
