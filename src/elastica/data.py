"""Turn raw transactions into a product x period panel ready for estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd

LOG_PRICE = "log_price"
LOG_QUANTITY = "log_quantity"
CALENDAR_FEATURES = ("trend", "month", "day_of_year", "week_sin", "week_cos")
DAILY_FEATURES = ("day_of_week",)
ALL_PRODUCTS = "all"


class DataError(ValueError):
    """The input cannot be used for estimation; the message says why."""


class Frequency(StrEnum):
    DAY = "D"
    WEEK = "W"
    MONTH = "M"


@dataclass(frozen=True, slots=True)
class Schema:
    """Which input columns hold each field. Only date, price and quantity are required."""

    date: str = "date"
    price: str = "price"
    quantity: str = "quantity"
    product: str | None = "product"
    cost: str | None = None
    controls: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Panel:
    frame: pd.DataFrame
    """One row per product and period: product, period, price, quantity, log columns, features."""
    features: tuple[str, ...]
    frequency: Frequency
    costs: dict[str, float] = field(default_factory=dict)
    dropped: int = 0
    """Periods dropped because their quantity or price was zero."""

    @property
    def products(self) -> list[str]:
        return sorted(self.frame["product"].unique())

    def for_product(self, product: str) -> pd.DataFrame:
        return self.frame[self.frame["product"] == product]


def read_transactions(path: str | Path) -> pd.DataFrame:
    file = Path(path)
    if not file.exists():
        raise DataError(f"{file} does not exist")
    if file.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(file)
        except ImportError as exc:
            raise DataError("reading Parquet needs pyarrow: pip install pyarrow") from exc
    return pd.read_csv(file)


def prepare(
    transactions: pd.DataFrame, schema: Schema | None = None, frequency: Frequency = Frequency.DAY
) -> Panel:
    """Validate, aggregate to product x period and add calendar features."""
    schema = schema or Schema()
    frame, controls = _encode_controls(_normalize(transactions, schema), schema.controls)
    costs = _costs(frame)
    aggregated = _aggregate(frame, controls, frequency)
    positive = (aggregated["quantity"] > 0) & (aggregated["price"] > 0)
    panel = aggregated[positive].copy()
    if panel.empty:
        raise DataError("no period has both a positive price and a positive quantity")
    calendar = _add_calendar(panel, frequency)
    panel[LOG_PRICE] = np.log(panel["price"])
    panel[LOG_QUANTITY] = np.log(panel["quantity"])
    return Panel(
        frame=panel.reset_index(drop=True),
        features=(*calendar, *controls),
        frequency=frequency,
        costs=costs,
        dropped=int((~positive).sum()),
    )


def _normalize(transactions: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    required = [schema.date, schema.price, schema.quantity, *schema.controls]
    optional = [c for c in (schema.product, schema.cost) if c is not None]
    missing = [c for c in [*required, *optional] if c not in transactions.columns]
    if missing:
        available = ", ".join(map(str, transactions.columns))
        raise DataError(f"missing column(s) {', '.join(missing)}; available: {available}")
    frame = pd.DataFrame(
        {
            "date": _dates(transactions[schema.date]),
            "price": _numbers(transactions[schema.price], schema.price),
            "quantity": _numbers(transactions[schema.quantity], schema.quantity),
            "product": transactions[schema.product].astype(str) if schema.product else ALL_PRODUCTS,
        }
    )
    if schema.cost:
        frame["cost"] = _numbers(transactions[schema.cost], schema.cost)
    for control in schema.controls:
        frame[control] = transactions[control].to_numpy()
    if (frame["price"] < 0).any() or (frame["quantity"] < 0).any():
        raise DataError("price and quantity must not be negative")
    return frame


def _dates(column: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(column, format="mixed")
    except (ValueError, TypeError) as exc:
        raise DataError(f"column {column.name!r} has values that are not dates") from exc


def _numbers(column: pd.Series, name: str) -> pd.Series:
    numbers = pd.to_numeric(column, errors="coerce")
    bad = numbers.isna() & column.notna()
    if bad.any():
        example = column[bad].iloc[0]
        raise DataError(f"column {name!r} has non-numeric values, e.g. {example!r}")
    if numbers.isna().any():
        raise DataError(f"column {name!r} has missing values")
    return numbers.astype(float)


def _encode_controls(
    frame: pd.DataFrame, controls: tuple[str, ...]
) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode non-numeric controls; return the frame and the final control columns."""
    names: list[str] = []
    for control in controls:
        if pd.api.types.is_numeric_dtype(frame[control]):
            names.append(control)
            continue
        dummies = pd.get_dummies(frame[control].astype(str), prefix=control, dtype=float)
        frame = pd.concat([frame.drop(columns=control), dummies], axis=1)
        names.extend(dummies.columns)
    return frame, names


def _costs(frame: pd.DataFrame) -> dict[str, float]:
    if "cost" not in frame.columns:
        return {}
    return {str(k): float(v) for k, v in frame.groupby("product")["cost"].median().items()}


def _aggregate(frame: pd.DataFrame, controls: list[str], frequency: Frequency) -> pd.DataFrame:
    frame = frame.assign(
        period=frame["date"].dt.to_period(frequency.value).dt.start_time,
        revenue=frame["price"] * frame["quantity"],
    )
    grouped = frame.groupby(["product", "period"], sort=True)
    result = grouped.agg(
        quantity=("quantity", "sum"),
        revenue=("revenue", "sum"),
        list_price=("price", "mean"),
        **{c: (c, "mean") for c in controls},
    ).reset_index()
    with np.errstate(divide="ignore", invalid="ignore"):
        weighted = result["revenue"] / result["quantity"]
    result["price"] = weighted.where(result["quantity"] > 0, result["list_price"])
    return result.drop(columns=["revenue", "list_price"])


def _add_calendar(panel: pd.DataFrame, frequency: Frequency) -> list[str]:
    period = panel["period"]
    start = period.min()
    panel["trend"] = (period - start).dt.days / 365.25
    panel["month"] = period.dt.month.astype(float)
    panel["day_of_year"] = period.dt.day_of_year.astype(float)
    week = period.dt.isocalendar().week.astype(float)
    panel["week_sin"] = np.sin(2 * np.pi * week / 52.18)
    panel["week_cos"] = np.cos(2 * np.pi * week / 52.18)
    features = list(CALENDAR_FEATURES)
    if frequency is Frequency.DAY:
        panel["day_of_week"] = period.dt.day_of_week.astype(float)
        features.extend(DAILY_FEATURES)
    return features
