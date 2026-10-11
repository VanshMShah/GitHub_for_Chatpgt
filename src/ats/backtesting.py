from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import chi2


def kupiec_pof(returns: np.ndarray, var_forecasts: np.ndarray, alpha: float = 0.01) -> dict[str, float | bool | str]:
    """Kupiec proportion-of-failures test for unconditional coverage."""
    y = np.asarray(returns, dtype=np.float64)
    v = np.asarray(var_forecasts, dtype=np.float64)
    if y.shape != v.shape:
        raise ValueError("returns and VaR forecasts must have equal length")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5")
    hits = (y < -v).astype(int)
    T = int(len(hits))
    N = int(hits.sum())
    p_hat = N / T if T else 0.0
    p0 = alpha

    def _safe_log(x: float) -> float:
        return float(np.log(max(x, 1e-12)))

    if N == 0 or N == T:
        lr = -2.0 * ((T - N) * _safe_log(1.0 - p0) + N * _safe_log(p0) - ((T - N) * _safe_log(1.0 - p_hat) + N * _safe_log(p_hat)))
    else:
        lr = -2.0 * ((T - N) * _safe_log(1.0 - p0) + N * _safe_log(p0) - (T - N) * _safe_log(1.0 - p_hat) - N * _safe_log(p_hat))
    p_value = float(chi2.sf(lr, 1))
    return {
        "observations": T,
        "breaches": N,
        "observed_failure_rate": float(p_hat),
        "target_failure_rate": p0,
        "lr_stat": float(lr),
        "p_value": p_value,
        "reject_h0": bool(p_value < 0.05),
        "decision": "Reject unconditional coverage" if p_value < 0.05 else "Accept unconditional coverage",
    }


def christoffersen_tests(returns: np.ndarray, var_forecasts: np.ndarray, alpha: float = 0.01) -> dict[str, Any]:
    """Christoffersen independence and joint conditional-coverage tests."""
    y = np.asarray(returns, dtype=np.float64)
    v = np.asarray(var_forecasts, dtype=np.float64)
    if y.shape != v.shape or len(y) < 3:
        raise ValueError("returns and VaR forecasts must have equal length >= 3")
    hits = (y < -v).astype(int)
    T = len(hits)

    n00 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 0)))
    n01 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 1)))
    n10 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 0)))
    n11 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 1)))

    p01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    p11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    p = (n01 + n11) / max(1, (T - 1))

    def term(count: int, prob: float) -> float:
        return 0.0 if count == 0 else count * np.log(max(prob, 1e-12))

    ll_ind_unrestricted = term(n00, 1 - p01) + term(n01, p01) + term(n10, 1 - p11) + term(n11, p11)
    ll_ind_restricted = term(n00, 1 - p) + term(n01, p) + term(n10, 1 - p) + term(n11, p)
    lr_ind = float(-2.0 * (ll_ind_restricted - ll_ind_unrestricted))
    p_ind = float(chi2.sf(max(lr_ind, 0.0), 1))

    pof = kupiec_pof(y, v, alpha=alpha)
    lr_cc = float(max(pof["lr_stat"] + lr_ind, 0.0))
    p_cc = float(chi2.sf(lr_cc, 2))

    return {
        "alpha": float(alpha),
        "transition_counts": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
        "transition_probabilities": {"p01": float(p01), "p11": float(p11)},
        "independence": {
            "lr_stat": lr_ind,
            "p_value": p_ind,
            "reject_h0": bool(p_ind < 0.05),
        },
        "conditional_coverage": {
            "lr_stat": lr_cc,
            "p_value": p_cc,
            "reject_h0": bool(p_cc < 0.05),
            "decision": "Reject conditional coverage" if p_cc < 0.05 else "Accept conditional coverage",
        },
    }


def backtest_var(returns: np.ndarray, var_forecasts: np.ndarray, alpha: float = 0.01) -> dict[str, Any]:
    pof = kupiec_pof(returns, var_forecasts, alpha=alpha)
    cc = christoffersen_tests(returns, var_forecasts, alpha=alpha)
    return {
        "summary": {
            "total_observations": int(len(returns)),
            "total_breaches": int(pof["breaches"]),
            "observed_failure_rate": float(pof["observed_failure_rate"]),
            "target_failure_rate": float(alpha),
        },
        "kupiec_pof": pof,
        "christoffersen_independence": cc["independence"],
        "christoffersen_conditional_coverage": cc["conditional_coverage"],
        "transition_counts": cc["transition_counts"],
        "transition_probabilities": cc["transition_probabilities"],
    }
