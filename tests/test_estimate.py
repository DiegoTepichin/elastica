import pandas as pd
import pytest

from elastica.data import Schema, prepare
from elastica.estimate import POOLED, EstimateOptions, estimate, estimates_frame
from elastica.simulate import SimulationConfig, simulate

SCHEMA = Schema(cost="unit_cost", controls=("promo",))


def test_estimates_each_product_close_to_truth() -> None:
    sim = simulate(SimulationConfig(n_products=3))
    estimates = estimate(prepare(sim.transactions, SCHEMA))

    costs = sim.transactions.groupby("product")["unit_cost"].first()
    assert [e.product for e in estimates] == ["SKU-001", "SKU-002", "SKU-003"]
    for item in estimates:
        assert item.result is not None
        assert item.result.elasticity == pytest.approx(sim.true_elasticity[item.product], abs=0.2)
        assert item.unit_cost == costs[item.product]
        assert item.current_price > 0
        assert item.current_quantity > 0


def test_pooled_estimate_with_product_fixed_effects() -> None:
    sim = simulate(SimulationConfig(n_products=3, elasticity_range=(-1.5, -1.5)))

    (pooled,) = estimate(prepare(sim.transactions, SCHEMA), pooled=True)

    assert pooled.product == POOLED
    assert pooled.result is not None
    assert pooled.result.elasticity == pytest.approx(-1.5, abs=0.15)


def test_products_with_too_little_data_report_an_error() -> None:
    sim = simulate(SimulationConfig(n_products=1, n_days=6))

    (item,) = estimate(prepare(sim.transactions, SCHEMA))

    assert item.result is None
    assert item.error is not None
    assert "at least 10" in item.error


def test_estimates_frame() -> None:
    sim = simulate(SimulationConfig(n_products=2, n_days=200))
    estimates = estimate(prepare(sim.transactions, SCHEMA), EstimateOptions(n_repeats=1))
    tiny = estimate(prepare(simulate(SimulationConfig(n_products=1, n_days=6)).transactions))

    frame = estimates_frame(estimates + tiny)

    assert list(frame["product"]) == ["SKU-001", "SKU-002", "SKU-001"]
    assert frame["n_obs"].tolist() == [200, 200, 0]
    assert pd.isna(frame["elasticity"].iloc[2])
    assert "at least" in frame["warnings"].iloc[2]
