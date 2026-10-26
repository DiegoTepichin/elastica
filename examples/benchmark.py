"""Compare naive OLS, OLS with linear controls and double ML on simulated data with known truth.

Usage: uv run --group docs python examples/benchmark.py [n_runs]
Writes docs/benchmark.svg and prints a Markdown table.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from elastica import Schema, SimulationConfig, estimate, prepare, simulate

COLORS = {"Naive OLS": "#eb6834", "OLS + linear controls": "#1baf7a", "elastica (DML)": "#2a78d6"}


@dataclass(frozen=True)
class Run:
    truth: float
    naive: float
    linear: float
    dml: float
    covered: bool


def run(seed: int) -> Run:
    rng = np.random.default_rng(1000 + seed)
    truth = round(float(rng.uniform(-2.5, -1.1)), 2)
    config = SimulationConfig(n_products=1, seed=seed, elasticity_range=(truth, truth))
    panel = prepare(simulate(config).transactions, Schema(controls=("promo",)))
    (item,) = estimate(panel)
    assert item.result is not None
    result = item.result
    return Run(
        truth,
        result.naive_elasticity,
        result.linear_elasticity,
        result.elasticity,
        result.ci_low <= truth <= result.ci_high,
    )


def main(n_runs: int) -> None:
    runs = [run(seed) for seed in range(n_runs)]
    truth = np.array([r.truth for r in runs])
    series = {
        "Naive OLS": np.array([r.naive for r in runs]),
        "OLS + linear controls": np.array([r.linear for r in runs]),
        "elastica (DML)": np.array([r.dml for r in runs]),
    }
    print(f"{n_runs} simulated products, true elasticity drawn from [-2.5, -1.1]\n")
    print("| Method | Mean absolute error | Mean bias |")
    print("|---|---|---|")
    for name, values in series.items():
        error = values - truth
        print(f"| {name} | {np.abs(error).mean():.2f} | {error.mean():+.2f} |")
    coverage = np.mean([r.covered for r in runs])
    print(f"\n95% confidence interval of elastica contains the truth in {coverage:.0%} of runs.")
    plot(truth, series, Path("docs/benchmark.svg"))


def plot(truth: np.ndarray, series: dict[str, np.ndarray], path: Path) -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "svg.fonttype": "none"})
    figure, axis = plt.subplots(figsize=(7.5, 5.2))
    low, high = -2.7, 1.0
    axis.plot(
        [low, high], [low, high], color="#8a8983", linewidth=1.2, linestyle="--", zorder=1,
        label="Perfect estimate",
    )  # fmt: skip
    for name, values in series.items():
        axis.scatter(
            truth, values, s=42, color=COLORS[name], edgecolor="#fcfcfb", linewidth=1.2,
            label=name, zorder=3,
        )  # fmt: skip
    axis.set_xlim(-2.6, -1.0)
    axis.set_ylim(low, high)
    axis.set_xlabel("True elasticity", color="#52514e")
    axis.set_ylabel("Estimated elasticity", color="#52514e")
    axis.set_title("Estimated vs true price elasticity, simulated data", loc="left", fontsize=13)
    axis.grid(color="#e8e7e2", linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color("#c9c8c2")
    axis.tick_params(colors="#52514e")
    axis.legend(frameon=False, loc="upper left")
    figure.patch.set_facecolor("#fcfcfb")
    axis.set_facecolor("#fcfcfb")
    figure.tight_layout()
    figure.savefig(path)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
