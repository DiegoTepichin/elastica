from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from elastica.cli import main
from elastica.data import DataError, Schema, prepare, read_transactions
from elastica.dml import estimate_elasticity
from elastica.estimate import EstimateOptions, estimate
from elastica.pricing import recommend
from elastica.simulate import SimulationConfig, simulate


def _sales(days: int = 120) -> pd.DataFrame:
    return simulate(SimulationConfig(n_products=1, n_days=days)).transactions


def _csv(tmp_path: Path, days: int = 120) -> Path:
    path = tmp_path / "sales.csv"
    _sales(days).to_csv(path, index=False)
    return path


def test_a_missing_date_is_an_error_not_a_dropped_row() -> None:
    sales = _sales()
    sales.loc[5, "date"] = pd.NaT

    with pytest.raises(DataError, match="column 'date' has missing values"):
        prepare(sales)


@pytest.mark.parametrize("column", ["price", "quantity", "unit_cost", "promo"])
def test_infinite_numbers_are_rejected_with_the_column_name(column: str) -> None:
    sales = _sales().astype({column: float})
    sales.loc[3, column] = np.inf
    schema = Schema(cost="unit_cost", controls=("promo",))

    with pytest.raises(DataError, match=f"column '{column}' has infinite values"):
        prepare(sales, schema)


def test_missing_numeric_controls_are_rejected() -> None:
    sales = _sales().astype({"promo": float})
    sales.loc[3, "promo"] = np.nan

    with pytest.raises(DataError, match="column 'promo' has missing values"):
        prepare(sales, Schema(controls=("promo",)))


def test_negative_costs_are_rejected() -> None:
    sales = _sales().assign(unit_cost=-1.0)

    with pytest.raises(DataError, match="must not be negative"):
        prepare(sales, Schema(cost="unit_cost"))


def test_a_directory_is_not_a_data_file(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="is not a file"):
        read_transactions(tmp_path)


BAD_OPTIONS = [
    ({"n_folds": 1}, "n_folds must be at least 2"),
    ({"n_repeats": 0}, "n_repeats must be at least 1"),
    ({"confidence": 0.0}, "confidence must be between 0 and 1"),
    ({"confidence": 1.0}, "confidence must be between 0 and 1"),
]


@pytest.mark.parametrize(("options", "message"), BAD_OPTIONS)
def test_options_are_validated(options: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EstimateOptions(**options)
    values = np.arange(40, dtype=float)
    with pytest.raises(ValueError, match=message):
        estimate_elasticity(values, values, values, **options)


@pytest.mark.parametrize("cost", [-5.0, float("nan"), float("inf")])
def test_unusable_unit_costs_are_rejected(cost: float) -> None:
    item = estimate(prepare(_sales()), EstimateOptions(n_repeats=1))[0]

    with pytest.raises(ValueError, match="unit cost must be a non-negative number"):
        recommend(item, unit_cost=cost)


@pytest.mark.parametrize("config", [{"n_products": 0}, {"n_days": 0}])
def test_empty_simulations_are_rejected(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        SimulationConfig(**config)


def test_cli_reports_a_directory_in_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["estimate", str(tmp_path)]) == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_cli_reports_an_unwritable_output_in_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "missing" / "out.csv"

    assert main(["estimate", str(_csv(tmp_path)), "--repeats", "1", "-o", str(output)]) == 1
    assert capsys.readouterr().err.splitlines()[-1].startswith("error: ")


@pytest.mark.parametrize("band", ["abc", "0.8", "0.8,1.2,1.4"])
def test_cli_explains_a_malformed_band(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], band: str
) -> None:
    assert main(["price", str(_csv(tmp_path)), "--band", band]) == 1
    assert "--band takes two numbers, such as 0.8,1.2" in capsys.readouterr().err
