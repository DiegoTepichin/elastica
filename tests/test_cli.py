from pathlib import Path

import pandas as pd
import pytest

from elastica import __version__
from elastica.cli import main


@pytest.fixture(scope="module")
def sales(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "sales.csv"
    assert main(["simulate", "-o", str(path), "--products", "2", "--days", "365"]) == 0
    return path


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--version"])

    assert __version__ in capsys.readouterr().out


def test_simulate_prints_the_truth(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "s.csv"

    assert main(["simulate", "-o", str(path), "--products", "2", "--days", "30"]) == 0

    assert len(pd.read_csv(path)) == 60
    assert "true elasticities: SKU-001 -2.2, SKU-002 -1.2" in capsys.readouterr().out


def test_estimate_prints_a_table_and_writes_csv(
    sales: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "estimates.csv"

    code = main(
        ["estimate", str(sales), "--controls", "promo", "--repeats", "1", "-o", str(output)]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "95% CI low" in out
    assert "SKU-002" in out
    assert pd.read_csv(output)["elasticity"].between(-3, -0.5).all()


def test_estimate_pooled_weekly(sales: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["estimate", str(sales), "--pooled", "--frequency", "W", "--repeats", "1"]

    assert main(args) == 0
    assert "(pooled)" in capsys.readouterr().out


def test_price_with_cost_column(
    sales: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "prices.csv"
    args = ["price", str(sales), "--controls", "promo", "--cost", "unit_cost", "--repeats", "1"]

    assert main([*args, "--band", "0.9,1.1", "-o", str(output)]) == 0

    frame = pd.read_csv(output)
    ratio = frame["recommended_price"] / frame["current_price"]
    assert ratio.between(0.899, 1.101).all()
    assert "Recommended" in capsys.readouterr().out


def test_price_with_flat_cost_and_single_product(
    sales: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["price", str(sales), "--product", "", "--unit-cost", "40", "--repeats", "1"]

    assert main(args) == 0
    assert "all" in capsys.readouterr().out


def test_reports_data_errors(sales: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["estimate", str(sales), "--price", "precio"]) == 1
    assert "error: missing column(s) precio" in capsys.readouterr().err


def test_reports_bad_band(sales: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["price", str(sales), "--band", "1.2,1.5"]) == 1
    assert "band" in capsys.readouterr().err
