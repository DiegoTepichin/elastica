import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from elastica.data import LOG_PRICE, LOG_QUANTITY, Schema, prepare
from elastica.dml import DMLResult, Folds, estimate_elasticity
from elastica.simulate import SimulationConfig, simulate


def simulated_result(n_days: int = 730, **kwargs: object) -> tuple[DMLResult, float]:
    sim = simulate(SimulationConfig(n_products=1, n_days=n_days, elasticity_range=(-1.8, -1.8)))
    panel = prepare(sim.transactions, Schema(controls=("promo",)))
    frame = panel.frame
    result = estimate_elasticity(
        frame[LOG_QUANTITY].to_numpy(),
        frame[LOG_PRICE].to_numpy(),
        frame[list(panel.features)].to_numpy(),
        **kwargs,  # type: ignore[arg-type]
    )
    return result, sim.true_elasticity["SKU-001"]


def test_recovers_the_true_elasticity() -> None:
    result, truth = simulated_result()

    assert result.elasticity == pytest.approx(truth, abs=0.15)
    assert result.ci_low < truth < result.ci_high
    assert result.p_value < 0.001
    assert result.warnings == ()


def test_naive_and_linear_estimates_are_biased_and_dml_is_not() -> None:
    result, truth = simulated_result()

    assert abs(result.naive_elasticity - truth) > 5 * abs(result.elasticity - truth)
    assert abs(result.linear_elasticity - truth) > 3 * abs(result.elasticity - truth)


def test_time_blocked_folds_with_three_seasons() -> None:
    result, truth = simulated_result(n_days=1095, folds=Folds.TIME, n_repeats=1)

    assert result.elasticity == pytest.approx(truth, abs=0.25)


def test_accepts_any_regressor() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 2))
    d = 0.5 * x[:, 0] + rng.normal(scale=0.3, size=400)
    y = -2.0 * d + x[:, 0] - x[:, 1] + rng.normal(scale=0.1, size=400)

    result = estimate_elasticity(y, d, x, learner=LinearRegression())

    assert result.elasticity == pytest.approx(-2.0, abs=0.05)
    assert result.linear_elasticity == pytest.approx(-2.0, abs=0.05)
    assert result.outcome_r2 > 0.5


def test_confidence_level_widens_interval() -> None:
    narrow, _ = simulated_result(confidence=0.8, n_repeats=1)
    wide, _ = simulated_result(confidence=0.99, n_repeats=1)

    assert wide.ci_high - wide.ci_low > narrow.ci_high - narrow.ci_low


def test_warns_on_weak_identification_and_positive_sign() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 1))
    d = x[:, 0] + rng.normal(scale=0.01, size=40)
    y = 0.5 * d + rng.normal(scale=0.1, size=40)

    result = estimate_elasticity(y, d, x, learner=LinearRegression(), n_repeats=1)

    assert len(result.warnings) == 3
    assert "only 40 observations" in result.warnings[0]
    assert "weakly identified" in result.warnings[1]
    assert "positive elasticity" in result.warnings[2]


@pytest.mark.parametrize(
    ("y", "d", "x", "message"),
    [
        (np.ones(5), np.ones(5), np.ones(5), "at least 10"),
        (np.ones(20), np.ones(19), np.ones(20), "same number of rows"),
        (np.ones(20), np.ones(20), np.ones(20), "price never changes"),
        (np.ones(20), np.r_[np.nan, np.arange(19.0)], np.ones(20), "NaN"),
    ],
)
def test_rejects_bad_input(y: np.ndarray, d: np.ndarray, x: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        estimate_elasticity(y, d, x)
