import numpy as np
import pandas as pd

from elastica.simulate import SimulationConfig, simulate


def test_shape_and_columns() -> None:
    sim = simulate(SimulationConfig(n_products=3, n_days=100))

    assert list(sim.transactions.columns) == [
        "date",
        "product",
        "price",
        "quantity",
        "promo",
        "unit_cost",
    ]
    assert len(sim.transactions) == 300
    assert sorted(sim.true_elasticity) == ["SKU-001", "SKU-002", "SKU-003"]


def test_elasticities_span_the_configured_range() -> None:
    sim = simulate(SimulationConfig(n_products=3, elasticity_range=(-2.0, -1.0)))

    assert list(sim.true_elasticity.values()) == [-2.0, -1.5, -1.0]


def test_is_reproducible() -> None:
    first = simulate(SimulationConfig(n_days=50)).transactions
    second = simulate(SimulationConfig(n_days=50)).transactions

    pd.testing.assert_frame_equal(first, second)


def test_price_and_demand_are_confounded_by_season() -> None:
    frame = simulate(SimulationConfig(n_products=1)).transactions
    month = frame["date"].dt.month
    summer = frame[month.isin([6, 7, 8])]
    winter = frame[month.isin([12, 1, 2])]

    assert summer["price"].mean() > winter["price"].mean()
    assert summer["quantity"].mean() > winter["quantity"].mean()


def test_naive_regression_is_biased_toward_zero() -> None:
    sim = simulate(SimulationConfig(n_products=1))
    frame = sim.transactions
    slope = np.polyfit(np.log(frame["price"]), np.log(frame["quantity"].clip(lower=1)), 1)[0]

    assert slope > sim.true_elasticity["SKU-001"] + 0.5
