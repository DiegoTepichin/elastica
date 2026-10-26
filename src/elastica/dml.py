"""Double machine learning for the partially linear model ``y = θ·d + g(X) + ε``.

With ``y = log quantity`` and ``d = log price``, ``θ`` is the price elasticity of demand.
Reference: Chernozhukov et al. (2018), "Double/debiased machine learning for treatment and
structural parameters", The Econometrics Journal 21(1).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from statistics import NormalDist
from typing import Protocol, Self

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

Array = NDArray[np.float64]

MIN_OBSERVATIONS = 60
WEAK_PRICE_SHARE = 0.05


class Regressor(Protocol):
    """Any scikit-learn style regressor."""

    def fit(self, x: Array, y: Array) -> Self: ...
    def predict(self, x: Array) -> Array: ...


class Folds(StrEnum):
    RANDOM = "random"
    """Shuffled folds, re-drawn on every repetition."""
    TIME = "time"
    """Contiguous blocks in time order, which limits leakage between neighbouring periods."""


@dataclass(frozen=True, slots=True)
class DMLResult:
    elasticity: float
    std_error: float
    ci_low: float
    ci_high: float
    confidence: float
    naive_elasticity: float
    """Slope of log quantity on log price with no controls."""
    linear_elasticity: float
    """Price coefficient of an OLS that adds the controls linearly."""
    n_obs: int
    outcome_r2: float
    """Cross-fitted R² of the model predicting log quantity from the controls."""
    price_r2: float
    """Cross-fitted R² of the model predicting log price from the controls."""
    warnings: tuple[str, ...]

    @property
    def residual_price_share(self) -> float:
        """Share of price variation left after removing what the controls explain."""
        return max(0.0, 1.0 - self.price_r2)

    @property
    def p_value(self) -> float:
        """Two-sided p-value for ``elasticity == 0``."""
        if self.std_error == 0:
            return 0.0
        return 2 * (1 - NormalDist().cdf(abs(self.elasticity / self.std_error)))


def default_learner(seed: int = 0) -> Regressor:
    model: Regressor = HistGradientBoostingRegressor(
        max_iter=100,
        learning_rate=0.1,
        max_leaf_nodes=15,
        min_samples_leaf=10,
        random_state=seed,
    )
    return model


def estimate_elasticity(
    y: Array,
    d: Array,
    x: Array,
    *,
    learner: Regressor | None = None,
    folds: Folds = Folds.RANDOM,
    n_folds: int = 5,
    n_repeats: int = 3,
    confidence: float = 0.95,
    seed: int = 0,
) -> DMLResult:
    """Estimate θ with cross-fitting. Rows must be in time order when ``folds`` is ``TIME``."""
    y, d, x = _check_inputs(y, d, x, n_folds)
    estimates: list[tuple[float, float]] = []
    r2_y: list[float] = []
    r2_d: list[float] = []
    for repeat in range(n_repeats):
        model = learner if learner is not None else default_learner(seed + repeat)
        splitter = _splitter(folds, n_folds, seed + repeat)
        y_hat = _cross_fit(model, x, y, splitter)
        d_hat = _cross_fit(model, x, d, splitter)
        estimates.append(_orthogonal_estimate(y - y_hat, d - d_hat))
        r2_y.append(_r2(y, y_hat))
        r2_d.append(_r2(d, d_hat))
    theta, std_error = _aggregate(estimates)
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    price_r2 = float(np.median(r2_d))
    return DMLResult(
        elasticity=theta,
        std_error=std_error,
        ci_low=theta - z * std_error,
        ci_high=theta + z * std_error,
        confidence=confidence,
        naive_elasticity=_ols_slope(y, d, None),
        linear_elasticity=_ols_slope(y, d, x),
        n_obs=len(y),
        outcome_r2=float(np.median(r2_y)),
        price_r2=price_r2,
        warnings=_warnings(theta, len(y), price_r2),
    )


def _check_inputs(y: Array, d: Array, x: Array, n_folds: int) -> tuple[Array, Array, Array]:
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if not (len(y) == len(d) == len(x)):
        raise ValueError("y, d and x must have the same number of rows")
    if len(y) < 2 * n_folds:
        raise ValueError(f"need at least {2 * n_folds} observations for {n_folds} folds")
    if not (np.isfinite(y).all() and np.isfinite(d).all() and np.isfinite(x).all()):
        raise ValueError("inputs contain NaN or infinite values")
    if np.ptp(d) == 0:
        raise ValueError("price never changes, so its effect cannot be estimated")
    return y, d, x


def _splitter(folds: Folds, n_folds: int, seed: int) -> KFold:
    if folds is Folds.TIME:
        return KFold(n_splits=n_folds, shuffle=False)
    return KFold(n_splits=n_folds, shuffle=True, random_state=seed)


def _cross_fit(model: Regressor, x: Array, target: Array, splitter: KFold) -> Array:
    predictions = np.empty_like(target)
    with threadpool_limits(limits=1, user_api="openmp"):
        for train, test in splitter.split(x):
            fitted: Regressor = clone(model).fit(x[train], target[train])
            predictions[test] = fitted.predict(x[test])
    return predictions


def _orthogonal_estimate(y_res: Array, d_res: Array) -> tuple[float, float]:
    """θ and its variance from the Neyman-orthogonal partialling-out score."""
    jacobian = float(np.mean(d_res**2))
    if jacobian == 0:
        raise ValueError("the controls explain price perfectly; no variation is left")
    theta = float(np.mean(d_res * y_res)) / jacobian
    score = (y_res - theta * d_res) * d_res
    variance = float(np.mean(score**2)) / jacobian**2 / len(y_res)
    return theta, variance


def _aggregate(estimates: list[tuple[float, float]]) -> tuple[float, float]:
    """Median over repetitions, with the spread between repetitions added to the variance."""
    thetas = np.array([t for t, _ in estimates])
    theta = float(np.median(thetas))
    variances = np.array([v for _, v in estimates]) + (thetas - theta) ** 2
    return theta, float(np.sqrt(np.median(variances)))


def _r2(target: Array, predicted: Array) -> float:
    total = float(np.sum((target - target.mean()) ** 2))
    if total == 0:
        return 0.0
    return 1 - float(np.sum((target - predicted) ** 2)) / total


def _ols_slope(y: Array, d: Array, x: Array | None) -> float:
    columns = [np.ones_like(d), d] if x is None else [np.ones_like(d), d, *x.T]
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coefficients[1])


def _warnings(theta: float, n_obs: int, price_r2: float) -> tuple[str, ...]:
    messages = []
    if n_obs < MIN_OBSERVATIONS:
        messages.append(f"only {n_obs} observations; the estimate is imprecise")
    if 1 - price_r2 < WEAK_PRICE_SHARE:
        messages.append(
            "the controls explain almost all price changes, so little independent price "
            "variation is left; the estimate is weakly identified"
        )
    if theta > 0:
        messages.append(
            "positive elasticity: demand rises with price, which usually means a confounder "
            "is missing from the controls"
        )
    return tuple(messages)
