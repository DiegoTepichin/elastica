"""Elasticity per product (or pooled across products) over a prepared panel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from elastica.data import LOG_PRICE, LOG_QUANTITY, Panel
from elastica.dml import DMLResult, Folds, Regressor, estimate_elasticity

POOLED = "(pooled)"
RECENT_PERIODS = 4


@dataclass(frozen=True, slots=True)
class EstimateOptions:
    learner: Regressor | None = None
    folds: Folds = Folds.RANDOM
    n_folds: int = 5
    n_repeats: int = 3
    confidence: float = 0.95
    seed: int = 0


@dataclass(frozen=True, slots=True)
class ProductEstimate:
    product: str
    result: DMLResult | None
    current_price: float
    """Median price over the most recent periods, the anchor for price recommendations."""
    current_quantity: float
    """Mean quantity per period over the same recent periods."""
    unit_cost: float | None
    error: str | None = None


def estimate(
    panel: Panel, options: EstimateOptions | None = None, *, pooled: bool = False
) -> list[ProductEstimate]:
    """One estimate per product, or a single fixed-effects estimate when ``pooled``."""
    options = options or EstimateOptions()
    if pooled:
        return [_pooled(panel, options)]
    return [_one(panel, product, options) for product in panel.products]


def estimates_frame(estimates: list[ProductEstimate]) -> pd.DataFrame:
    rows = []
    for item in estimates:
        result = item.result
        rows.append(
            {
                "product": item.product,
                "elasticity": result.elasticity if result else np.nan,
                "ci_low": result.ci_low if result else np.nan,
                "ci_high": result.ci_high if result else np.nan,
                "std_error": result.std_error if result else np.nan,
                "naive_elasticity": result.naive_elasticity if result else np.nan,
                "linear_elasticity": result.linear_elasticity if result else np.nan,
                "n_obs": result.n_obs if result else 0,
                "price_r2": result.price_r2 if result else np.nan,
                "current_price": item.current_price,
                "unit_cost": item.unit_cost,
                "warnings": "; ".join(result.warnings) if result else item.error,
            }
        )
    return pd.DataFrame(rows)


def _one(panel: Panel, product: str, options: EstimateOptions) -> ProductEstimate:
    frame = panel.for_product(product).sort_values("period")
    return _fit(frame, list(panel.features), product, panel.costs.get(product), options)


def _pooled(panel: Panel, options: EstimateOptions) -> ProductEstimate:
    frame = panel.frame.sort_values("period")
    dummies = pd.get_dummies(frame["product"], prefix="product", dtype=float, drop_first=True)
    frame = pd.concat([frame, dummies], axis=1)
    features = [*panel.features, *dummies.columns]
    costs = list(panel.costs.values())
    cost = float(np.median(costs)) if costs else None
    return _fit(frame, features, POOLED, cost, options)


def _fit(
    frame: pd.DataFrame,
    features: list[str],
    product: str,
    cost: float | None,
    options: EstimateOptions,
) -> ProductEstimate:
    recent = frame[
        frame["period"] >= frame["period"].drop_duplicates().nlargest(RECENT_PERIODS).min()
    ]
    current_price = float(recent["price"].median())
    current_quantity = float(recent.groupby("product")["quantity"].mean().sum())
    try:
        result = estimate_elasticity(
            frame[LOG_QUANTITY].to_numpy(),
            frame[LOG_PRICE].to_numpy(),
            frame[features].to_numpy(dtype=float),
            learner=options.learner,
            folds=options.folds,
            n_folds=options.n_folds,
            n_repeats=options.n_repeats,
            confidence=options.confidence,
            seed=options.seed,
        )
    except ValueError as exc:
        return ProductEstimate(product, None, current_price, current_quantity, cost, str(exc))
    return ProductEstimate(product, result, current_price, current_quantity, cost)
