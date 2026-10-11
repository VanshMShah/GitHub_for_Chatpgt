from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.arima.model import ARIMA


def fit_arima_mean_model(
    returns: np.ndarray,
    *,
    max_p: int = 2,
    max_q: int = 2,
    criterion: str = "aic",
) -> dict[str, Any]:
    """Fit ARIMA(p,0,q) models and return the best fitted mean plus residuals."""
    y = np.asarray(returns, dtype=np.float64)
    if y.ndim != 1 or len(y) < 40:
        raise ValueError("returns must be one-dimensional with at least 40 observations")
    if criterion not in {"aic", "bic"}:
        raise ValueError("criterion must be 'aic' or 'bic'")

    best_value = np.inf
    best_order: tuple[int, int, int] | None = None
    best_result = None

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    fitted = ARIMA(
                        y,
                        order=(p, 0, q),
                        trend="c",
                        enforce_stationarity=True,
                        enforce_invertibility=True,
                    ).fit()
                value = float(fitted.aic if criterion == "aic" else fitted.bic)
                if np.isfinite(value) and value < best_value:
                    best_value = value
                    best_order = (p, 0, q)
                    best_result = fitted
            except Exception:
                continue

    if best_result is None or best_order is None:
        raise RuntimeError("ARIMA grid search failed to converge")

    residuals = np.asarray(best_result.resid, dtype=np.float64)
    fitted_mean = np.asarray(best_result.fittedvalues, dtype=np.float64)
    forecast = np.asarray(best_result.get_forecast(steps=1).predicted_mean, dtype=np.float64)

    # ARCH-LM is a diagnostic gate for variance-model usefulness, not a proof of GARCH validity.
    arch_lm_stat, arch_lm_pvalue, f_stat, f_pvalue = het_arch(residuals, nlags=min(10, max(3, len(residuals) // 10)))
    arch_effects_present = bool(arch_lm_pvalue < 0.05)

    params = {name: float(value) for name, value in zip(best_result.param_names, best_result.params)}

    return {
        "order": best_order,
        "ic_type": criterion.upper(),
        "ic_value": float(best_value),
        "params": params,
        "fitted_mean": fitted_mean,
        "residuals": residuals,
        "mean_forecast": float(forecast[0]),
        "arch_lm_test": {
            "lm_statistic": float(arch_lm_stat),
            "lm_p_value": float(arch_lm_pvalue),
            "f_statistic": float(f_stat),
            "f_p_value": float(f_pvalue),
            "arch_effects_present": arch_effects_present,
            "decision": "Proceed to GJR-GARCH volatility modeling" if arch_effects_present else "Weak ARCH evidence; variance modeling may be unnecessary",
        },
    }
