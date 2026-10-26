"""Synthetic daily sales with a known price elasticity and realistic confounding.

Demand and price both rise in high season and around two sharp sales peaks (mid-November and
Christmas), so a naive regression of quantity on price underestimates how price-sensitive
customers are. Season and peaks are observable from the date, which is what lets double machine
learning recover the true elasticity; the peaks are non-linear in the calendar, which is what
defeats a linear regression with the same controls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class Simulation:
    transactions: pd.DataFrame
    """Columns: date, product, price, quantity, promo, unit_cost."""
    true_elasticity: dict[str, float]


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    n_products: int = 4
    n_days: int = 730
    start: str = "2024-01-01"
    elasticity_range: tuple[float, float] = (-2.2, -1.2)
    base_price: float = 100.0
    base_quantity: float = 60.0
    season_on_price: float = 0.15
    season_on_demand: float = 0.6
    peak_on_price: float = 0.2
    peak_on_demand: float = 0.8
    promo_rate: float = 0.12
    promo_discount: float = 0.2
    promo_lift: float = 0.3
    price_noise: float = 0.08
    demand_noise: float = 0.15
    margin: float = 0.55
    seed: int = 7


def simulate(config: SimulationConfig | None = None) -> Simulation:
    cfg = config or SimulationConfig()
    rng = np.random.default_rng(cfg.seed)
    dates = pd.date_range(cfg.start, periods=cfg.n_days, freq="D")
    day_of_year = dates.day_of_year.to_numpy()
    season = np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    weekend = (dates.day_of_week.to_numpy() >= 5).astype(float)
    peak = _peaks(day_of_year)

    low, high = cfg.elasticity_range
    elasticities = np.round(np.linspace(low, high, cfg.n_products), 2)
    frames = []
    truth: dict[str, float] = {}
    for index, elasticity in enumerate(elasticities):
        product = f"SKU-{index + 1:03d}"
        truth[product] = float(elasticity)
        base_price = cfg.base_price * rng.uniform(0.6, 1.6)
        promo = rng.random(cfg.n_days) < cfg.promo_rate
        log_price = (
            np.log(base_price)
            + cfg.season_on_price * season
            + cfg.peak_on_price * peak
            - cfg.promo_discount * promo
            + rng.normal(0, cfg.price_noise, cfg.n_days)
        )
        log_demand = (
            np.log(cfg.base_quantity)
            + elasticity * (log_price - np.log(base_price))
            + cfg.season_on_demand * season
            + cfg.peak_on_demand * peak
            + cfg.promo_lift * promo
            + 0.25 * weekend
            + rng.normal(0, cfg.demand_noise, cfg.n_days)
        )
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "product": product,
                    "price": np.round(np.exp(log_price), 2),
                    "quantity": rng.poisson(np.exp(log_demand)),
                    "promo": promo.astype(int),
                    "unit_cost": round(base_price * (1 - cfg.margin), 2),
                }
            )
        )
    return Simulation(pd.concat(frames, ignore_index=True), truth)


def _peaks(day_of_year: np.ndarray) -> np.ndarray:
    """Narrow bumps (about a week wide) centred on mid-November and Christmas."""
    centres = (320, 357)
    return np.sum([np.exp(-0.5 * ((day_of_year - c) / 5.0) ** 2) for c in centres], axis=0)
