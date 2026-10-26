"""Causal price elasticity with double machine learning."""

from importlib.metadata import version

from elastica.data import DataError, Frequency, Panel, Schema, prepare, read_transactions
from elastica.dml import DMLResult, Folds, estimate_elasticity
from elastica.estimate import EstimateOptions, ProductEstimate, estimate, estimates_frame
from elastica.pricing import (
    PriceBand,
    Recommendation,
    demand_curve,
    optimal_price,
    recommend,
    recommendations_frame,
)
from elastica.simulate import Simulation, SimulationConfig, simulate

__version__ = version("elastica")

__all__ = [
    "DMLResult",
    "DataError",
    "EstimateOptions",
    "Folds",
    "Frequency",
    "Panel",
    "PriceBand",
    "ProductEstimate",
    "Recommendation",
    "Schema",
    "Simulation",
    "SimulationConfig",
    "__version__",
    "demand_curve",
    "estimate",
    "estimate_elasticity",
    "estimates_frame",
    "optimal_price",
    "prepare",
    "read_transactions",
    "recommend",
    "recommendations_frame",
    "simulate",
]
