from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from elastica.data import (
    ALL_PRODUCTS,
    DataError,
    Frequency,
    Schema,
    prepare,
    read_transactions,
)
from elastica.simulate import SimulationConfig, simulate


def small_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fecha": ["2026-01-05", "2026-01-05", "2026-01-06", "2026-01-12"],
            "sku": ["A", "A", "A", "B"],
            "precio": [10.0, 12.0, 11.0, 5.0],
            "unidades": [1, 3, 0, 4],
            "canal": ["web", "tienda", "web", "web"],
            "costo": [6.0, 6.0, 6.0, 2.0],
        }
    )


SCHEMA = Schema(
    date="fecha",
    price="precio",
    quantity="unidades",
    product="sku",
    cost="costo",
    controls=("canal",),
)


def test_aggregates_to_quantity_weighted_price() -> None:
    panel = prepare(small_frame(), SCHEMA)
    first = panel.for_product("A").iloc[0]

    assert first["quantity"] == 4
    assert first["price"] == pytest.approx((10 * 1 + 12 * 3) / 4)
    assert first[["canal_tienda", "canal_web"]].tolist() == [0.5, 0.5]


def test_drops_periods_without_sales() -> None:
    panel = prepare(small_frame(), SCHEMA)

    assert panel.dropped == 1
    assert len(panel.frame) == 2
    assert panel.products == ["A", "B"]


def test_weekly_frequency_merges_days() -> None:
    panel = prepare(small_frame(), SCHEMA, Frequency.WEEK)

    assert panel.for_product("A")["quantity"].tolist() == [4]
    assert "day_of_week" not in panel.features


def test_features_include_calendar_and_controls() -> None:
    panel = prepare(small_frame(), SCHEMA)

    assert panel.features == (
        "trend",
        "month",
        "week_sin",
        "week_cos",
        "day_of_week",
        "canal_tienda",
        "canal_web",
    )


def test_log_columns_and_costs() -> None:
    panel = prepare(small_frame(), SCHEMA)

    assert np.allclose(panel.frame["log_quantity"], np.log(panel.frame["quantity"]))
    assert panel.costs == {"A": 6.0, "B": 2.0}


def test_without_product_column_everything_is_one_product() -> None:
    schema = Schema(date="fecha", price="precio", quantity="unidades", product=None)

    assert prepare(small_frame(), schema).products == [ALL_PRODUCTS]


def test_simulated_data_prepares_with_default_schema() -> None:
    sim = simulate(SimulationConfig(n_products=2, n_days=60))
    panel = prepare(sim.transactions, Schema(cost="unit_cost", controls=("promo",)))

    assert panel.products == ["SKU-001", "SKU-002"]
    assert "promo" in panel.features


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda f: f.drop(columns="precio"), "missing column"),
        (lambda f: f.assign(precio=["10", "x", "1", "2"]), "non-numeric values, e.g. 'x'"),
        (lambda f: f.assign(unidades=[1, None, 1, 1]), "missing values"),
        (lambda f: f.assign(unidades=[1, -1, 1, 1]), "must not be negative"),
        (lambda f: f.assign(fecha=["nope"] * 4), "not dates"),
        (lambda f: f.assign(unidades=[0, 0, 0, 0]), "no period has both"),
    ],
)
def test_reports_bad_input(change: object, message: str) -> None:
    frame = change(small_frame())  # type: ignore[operator]

    with pytest.raises(DataError, match=message):
        prepare(frame, SCHEMA)


def test_read_transactions(tmp_path: Path) -> None:
    path = tmp_path / "sales.csv"
    small_frame().to_csv(path, index=False)

    assert len(read_transactions(path)) == 4
    with pytest.raises(DataError, match="does not exist"):
        read_transactions(tmp_path / "missing.csv")


@pytest.mark.skipif(find_spec("pyarrow") is not None, reason="pyarrow is installed")
def test_parquet_without_pyarrow_explains_what_to_install(tmp_path: Path) -> None:
    path = tmp_path / "sales.parquet"
    path.write_bytes(b"PAR1")

    with pytest.raises(DataError, match="needs pyarrow"):
        read_transactions(path)
