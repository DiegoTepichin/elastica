"""Turn an elasticity into a price, under explicit and conservative assumptions.

Demand is modelled as constant-elasticity around the current price, ``q = q0 · (p / p0)^θ``.
That approximation is only trustworthy near prices that were actually observed, so every
recommendation is kept inside a band around the current price.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from elastica.estimate import ProductEstimate


class Objective(StrEnum):
    PROFIT = "profit"
    REVENUE = "revenue"


class Reason(StrEnum):
    OPTIMUM = "optimum inside the band"
    UPPER = "capped at the top of the band"
    LOWER = "capped at the bottom of the band"
    NO_ESTIMATE = "no elasticity estimate"


@dataclass(frozen=True, slots=True)
class PriceBand:
    """Multipliers of the current price that bound any recommendation."""

    lower: float = 0.8
    upper: float = 1.2

    def __post_init__(self) -> None:
        if not 0 < self.lower <= 1 <= self.upper:
            raise ValueError("the band must satisfy 0 < lower <= 1 <= upper")


@dataclass(frozen=True, slots=True)
class Recommendation:
    product: str
    objective: Objective
    current_price: float
    recommended_price: float
    price_range: tuple[float, float]
    """Recommended price at each end of the elasticity's confidence interval."""
    elasticity: float | None
    unit_cost: float | None
    quantity_change: float
    revenue_change: float
    profit_change: float | None
    reason: Reason
    warnings: tuple[str, ...]


def optimal_price(elasticity: float, unit_cost: float) -> float | None:
    """Profit-maximizing price ``c · θ / (1 + θ)``; None when demand is inelastic (θ ≥ -1)."""
    if elasticity >= -1:
        return None
    return unit_cost * elasticity / (1 + elasticity)


def recommend(
    item: ProductEstimate, *, unit_cost: float | None = None, band: PriceBand | None = None
) -> Recommendation:
    band = band or PriceBand()
    cost = unit_cost if unit_cost is not None else item.unit_cost
    objective = Objective.PROFIT if cost is not None else Objective.REVENUE
    current = item.current_price
    if item.result is None:
        return Recommendation(
            item.product, objective, current, current, (current, current), None, cost,
            0.0, 0.0, 0.0 if cost is not None else None, Reason.NO_ESTIMATE,
            (item.error or "no estimate",),
        )  # fmt: skip
    theta = item.result.elasticity
    price, reason = _best_price(theta, cost, current, band)
    ends = sorted(
        _best_price(bound, cost, current, band)[0]
        for bound in (item.result.ci_low, item.result.ci_high)
    )
    ratio = price / current
    quantity_ratio = ratio**theta
    profit_change = None
    if cost is not None and current != cost:
        profit_change = (price - cost) * quantity_ratio / (current - cost) - 1
    return Recommendation(
        product=item.product,
        objective=objective,
        current_price=current,
        recommended_price=round(price, 2),
        price_range=(round(ends[0], 2), round(ends[1], 2)),
        elasticity=theta,
        unit_cost=cost,
        quantity_change=quantity_ratio - 1,
        revenue_change=ratio * quantity_ratio - 1,
        profit_change=profit_change,
        reason=reason,
        warnings=_warnings(item, cost, reason),
    )


def demand_curve(
    current_price: float,
    current_quantity: float,
    elasticity: float,
    *,
    unit_cost: float | None = None,
    band: PriceBand | None = None,
    points: int = 41,
) -> pd.DataFrame:
    """Price, quantity, revenue and (with a cost) profit across the band."""
    band = band or PriceBand()
    prices = np.linspace(band.lower * current_price, band.upper * current_price, points)
    quantity = current_quantity * (prices / current_price) ** elasticity
    frame = pd.DataFrame({"price": prices, "quantity": quantity, "revenue": prices * quantity})
    if unit_cost is not None:
        frame["profit"] = (prices - unit_cost) * quantity
    return frame


def recommendations_frame(recommendations: list[Recommendation]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product": r.product,
                "objective": r.objective.value,
                "elasticity": r.elasticity,
                "unit_cost": r.unit_cost,
                "current_price": r.current_price,
                "recommended_price": r.recommended_price,
                "price_low": r.price_range[0],
                "price_high": r.price_range[1],
                "quantity_change": r.quantity_change,
                "revenue_change": r.revenue_change,
                "profit_change": r.profit_change,
                "reason": r.reason.value,
                "warnings": "; ".join(r.warnings),
            }
            for r in recommendations
        ]
    )


def _best_price(
    elasticity: float, cost: float | None, current: float, band: PriceBand
) -> tuple[float, Reason]:
    low, high = band.lower * current, band.upper * current
    if cost is None:
        # Revenue p·q grows with price when θ > -1 and falls when θ < -1.
        return (high, Reason.UPPER) if elasticity > -1 else (low, Reason.LOWER)
    target = optimal_price(elasticity, cost)
    if target is None or target > high:
        return high, Reason.UPPER
    if target < low:
        return low, Reason.LOWER
    return target, Reason.OPTIMUM


def _warnings(item: ProductEstimate, cost: float | None, reason: Reason) -> tuple[str, ...]:
    assert item.result is not None
    messages = list(item.result.warnings)
    if item.result.elasticity >= -1:
        messages.append(
            "demand looks inelastic, so the model keeps pushing the price up; it is capped at "
            "the band because the constant-elasticity assumption breaks down far from observed "
            "prices"
        )
    if cost is not None and item.current_price <= cost:
        messages.append("the current price does not cover the unit cost")
    if cost is None:
        messages.append("no unit cost given: maximizing revenue, not profit")
    if item.result.ci_low < -1 < item.result.ci_high:
        messages.append("the confidence interval includes -1; elastic vs inelastic is uncertain")
    if reason is not Reason.OPTIMUM and cost is not None:
        messages.append(f"price {reason.value}; consider testing a wider band in small steps")
    return tuple(messages)
