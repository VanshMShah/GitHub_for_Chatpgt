from __future__ import annotations

from math import lgamma, log, pi
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import t as student_t


def hansen_skewed_t_constants(nu: float, lam: float) -> tuple[float, float, float]:
    if nu <= 2.0:
        raise ValueError("nu must be > 2")
    if not (-0.999 < lam < 0.999):
        raise ValueError("lambda must be in (-0.999, 0.999)")
    c = np.exp(gammaln((nu + 1.0) / 2.0) - 0.5 * log(pi * (nu - 2.0)) - gammaln(nu / 2.0))
    a = 4.0 * lam * c * (nu - 2.0) / (nu - 1.0)
    b2 = 1.0 + 3.0 * lam * lam - a * a
    if b2 <= 0:
        raise ValueError("invalid Hansen skew-t parameters: b^2 <= 0")
    b = float(np.sqrt(b2))
    return float(a), b, float(c)


def hansen_skewed_t_logpdf(z: np.ndarray | float, nu: float, lam: float) -> np.ndarray:
    z_arr = np.asarray(z, dtype=np.float64)
    a, b, c = hansen_skewed_t_constants(nu, lam)
    threshold = -a / b
    out = np.empty_like(z_arr, dtype=np.float64)

    left = z_arr < threshold
    term_const = log(b) + log(c)
    out[left] = term_const - ((nu + 1.0) / 2.0) * np.log1p(((b * z_arr[left] + a) / (1.0 - lam)) ** 2 / (nu - 2.0))
    out[~left] = term_const - ((nu + 1.0) / 2.0) * np.log1p(((b * z_arr[~left] + a) / (1.0 + lam)) ** 2 / (nu - 2.0))
    return out


def hansen_skewed_t_ppf(q: float, nu: float, lam: float) -> float:
    """Analytic quantile for Hansen's standardized skewed-t parameterization."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    a, b, _ = hansen_skewed_t_constants(nu, lam)
    q_cut = (1.0 - lam) / 2.0
    scale = np.sqrt((nu - 2.0) / nu)
    if q < q_cut:
        y = float(student_t.ppf(q / (1.0 - lam), df=nu))
        u = y * scale
        return float(((1.0 - lam) * u - a) / b)
    y = float(student_t.ppf(0.5 + (q - q_cut) / (1.0 + lam), df=nu))
    u = y * scale
    return float(((1.0 + lam) * u - a) / b)


def hansen_skewed_t_cdf(z: float, nu: float, lam: float) -> float:
    a, b, _ = hansen_skewed_t_constants(nu, lam)
    scale = np.sqrt((nu - 2.0) / nu)
    if z < -a / b:
        return float((1.0 - lam) * student_t.cdf((b*z+a) / ((1.0-lam)*scale), df=nu))
    return float((1.0-lam)/2.0 + (1.0+lam) * (student_t.cdf((b*z+a)/((1.0+lam)*scale), df=nu)-0.5))


def _variance_filter(residuals: np.ndarray, omega: float, alpha: float, gamma: float, beta: float) -> np.ndarray:
    eps = np.asarray(residuals, dtype=np.float64)
    sigma2 = np.empty(len(eps), dtype=np.float64)
    sigma2[0] = max(float(np.var(eps)), 1e-8)
    for i in range(1, len(eps)):
        prev = eps[i - 1]
        leverage = gamma if prev < 0 else 0.0
        sigma2[i] = max(1e-10, omega + (alpha + leverage) * prev * prev + beta * sigma2[i - 1])
    return sigma2


def _neg_log_likelihood(theta: np.ndarray, residuals: np.ndarray) -> float:
    omega, alpha, gamma, beta, nu, lam = theta
    if omega <= 0 or alpha < 0 or gamma < 0 or beta < 0 or nu <= 2.0 or abs(lam) >= 0.999:
        return 1e100
    if alpha + beta + 0.5 * gamma >= 0.999:
        return 1e100
    try:
        sigma2 = _variance_filter(residuals, omega, alpha, gamma, beta)
        z = residuals / np.sqrt(sigma2)
        logpdf = hansen_skewed_t_logpdf(z, nu, lam) - 0.5 * np.log(sigma2)
        if not np.all(np.isfinite(logpdf)):
            return 1e100
        return float(-np.sum(logpdf))
    except Exception:
        return 1e100


def fit_gjr_garch(residuals: np.ndarray) -> dict[str, Any]:
    """Fit GJR-GARCH(1,1) with Hansen skewed-t innovations."""
    eps = np.asarray(residuals, dtype=np.float64)
    if eps.ndim != 1 or len(eps) < 60:
        raise ValueError("residuals must be one-dimensional with at least 60 observations")
    if not np.all(np.isfinite(eps)):
        raise ValueError("residuals must be finite")

    sample_var = float(np.var(eps))
    x0 = np.array([max(sample_var * 0.05, 1e-5), 0.05, 0.05, 0.85, 8.0, 0.0], dtype=float)
    bounds = [
        (1e-10, max(sample_var * 2.0, 1.0)),
        (1e-8, 0.5),
        (1e-8, 0.9),
        (1e-8, 0.98),
        (2.01, 50.0),
        (-0.95, 0.95),
    ]
    constraints = [{"type": "ineq", "fun": lambda x: 0.999 - x[0] * 0 - x[1] - x[2] * 0.5 - x[3]}]

    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(eps,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 800, "ftol": 1e-8, "disp": False},
    )

    if not result.success or not np.isfinite(result.fun):
        # A second seed is useful on flat likelihood surfaces.
        alt_x0 = np.array([max(sample_var * 0.02, 1e-5), 0.03, 0.08, 0.82, 12.0, -0.1], dtype=float)
        result = minimize(
            _neg_log_likelihood,
            alt_x0,
            args=(eps,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-8, "disp": False},
        )

    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"GJR-GARCH MLE failed: {result.message}")

    omega, alpha, gamma, beta, nu, lam = map(float, result.x)
    sigma2 = _variance_filter(eps, omega, alpha, gamma, beta)
    z = eps / np.sqrt(sigma2)
    persistence = alpha + beta + 0.5 * gamma

    return {
        "params": {
            "omega": omega,
            "alpha": alpha,
            "gamma": gamma,
            "beta": beta,
            "nu": nu,
            "lambda": lam,
        },
        "persistence": float(persistence),
        "log_likelihood": float(-result.fun),
        "sigma2": sigma2,
        "standardized_residuals": z,
        "optimization": {"success": bool(result.success), "message": str(result.message), "iterations": int(result.nit)},
    }


def one_step_variance_forecast(residuals: np.ndarray, garch_result: dict[str, Any]) -> float:
    p = garch_result["params"]
    eps_last = float(np.asarray(residuals, dtype=np.float64)[-1])
    sigma2_last = float(np.asarray(garch_result["sigma2"], dtype=np.float64)[-1])
    leverage = p["gamma"] if eps_last < 0 else 0.0
    return float(max(1e-10, p["omega"] + (p["alpha"] + leverage) * eps_last**2 + p["beta"] * sigma2_last))
