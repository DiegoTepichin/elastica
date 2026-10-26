import math

import numpy as np
import pytest

from elastica.dml import DMLResult
from elastica.estimate import ProductEstimate
from elastica.pricing import (
    Objective,
    PriceBand,
    Reason,
    demand_curve,
    optimal_price,
    recommend,
    recommendations_frame,
)


def result(theta: float, half_width: float = 0.1, warnings: tuple[str, ...] = ()) -> DMLResult:
    return DMLResult(
        elasticity=theta,
        std_error=half_width / 1.96,
        ci_low=theta - half_width,
        ci_high=theta + half_width,
        confidence=0.95,
        naive_elasticity=theta / 2,
        linear_elasticity=theta,
        n_obs=500,
        outcome_r2=0.6,
        price_r2=0.4,
        warnings=warnings,
    )


def item(theta: float | None, cost: float | None = 60.0, price: float = 100.0) -> ProductEstimate:
    return ProductEstimate(
        product="SKU-001",
        result=None if theta is None else result(theta),
        current_price=price,
        current_quantity=50.0,
        unit_cost=cost,
        error=None if theta is not None else "too little data",
    )


@pytest.mark.parametrize(
    ("theta", "cost", "expected"),
    [(-2.0, 50.0, 100.0), (-3.0, 60.0, 90.0), (-1.0, 50.0, None), (-0.5, 50.0, None)],
)
def test_optimal_price_follows_the_lerner_rule(
    theta: float, cost: float, expected: float | None
) -> None:
    assert optimal_price(theta, cost) == (pytest.approx(expected) if expected else None)


def test_interior_optimum() -> None:
    rec = recommend(item(-3.0))

    assert rec.reason is Reason.OPTIMUM
    assert rec.recommended_price == 90.0
    assert rec.objective is Objective.PROFIT
    assert rec.quantity_change == pytest.approx(0.9**-3 - 1)
    assert rec.profit_change is not None
    assert rec.profit_change > 0
    assert rec.price_range[0] <= rec.recommended_price <= rec.price_range[1]


def test_optimum_is_profit_maximizing_on_the_curve() -> None:
    rec = recommend(item(-2.5), band=PriceBand(0.5, 1.5))
    curve = demand_curve(100.0, 50.0, -2.5, unit_cost=60.0, band=PriceBand(0.5, 1.5), points=1001)

    best = curve["price"].iloc[int(np.argmax(curve["profit"].to_numpy()))]
    assert rec.recommended_price == pytest.approx(best, abs=0.2)


def test_inelastic_demand_is_capped_with_a_warning() -> None:
    rec = recommend(item(-0.6))

    assert rec.recommended_price == 120.0
    assert rec.reason is Reason.UPPER
    assert any("inelastic" in w for w in rec.warnings)


def test_very_elastic_demand_hits_the_lower_band() -> None:
    rec = recommend(item(-8.0, cost=50.0))

    assert rec.reason is Reason.LOWER
    assert rec.recommended_price == 80.0


def test_without_cost_maximizes_revenue() -> None:
    elastic = recommend(item(-2.0, cost=None))
    inelastic = recommend(item(-0.5, cost=None))

    assert elastic.objective is Objective.REVENUE
    assert elastic.recommended_price == 80.0
    assert inelastic.recommended_price == 120.0
    assert elastic.profit_change is None
    assert any("no unit cost" in w for w in elastic.warnings)


def test_explicit_cost_overrides_the_estimate() -> None:
    assert recommend(item(-3.0, cost=None), unit_cost=60.0).recommended_price == 90.0


def test_uncertain_sign_and_cost_warnings() -> None:
    uncertain = ProductEstimate("A", result(-1.1, half_width=0.5), 50.0, 10.0, 60.0)

    rec = recommend(uncertain)

    assert any("includes -1" in w for w in rec.warnings)
    assert any("does not cover the unit cost" in w for w in rec.warnings)


def test_missing_estimate_keeps_the_current_price() -> None:
    rec = recommend(item(None))

    assert rec.reason is Reason.NO_ESTIMATE
    assert rec.recommended_price == rec.current_price
    assert rec.warnings == ("too little data",)


def test_band_validation() -> None:
    with pytest.raises(ValueError, match="band"):
        PriceBand(1.1, 1.3)


def test_demand_curve_without_cost() -> None:
    curve = demand_curve(10.0, 100.0, -2.0, points=5)

    assert list(curve.columns) == ["price", "quantity", "revenue"]
    assert curve["quantity"].iloc[0] == pytest.approx(100 * 0.8**-2)


def test_recommendations_frame() -> None:
    frame = recommendations_frame([recommend(item(-3.0)), recommend(item(None))])

    assert frame["recommended_price"].tolist() == [90.0, 100.0]
    assert frame["reason"].tolist() == [Reason.OPTIMUM.value, Reason.NO_ESTIMATE.value]
    assert math.isnan(frame["elasticity"].iloc[1])
