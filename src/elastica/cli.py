"""`elastica` command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from elastica import __version__
from elastica.data import DataError, Frequency, Schema, prepare, read_transactions
from elastica.dml import Folds
from elastica.estimate import EstimateOptions, ProductEstimate, estimate, estimates_frame
from elastica.pricing import PriceBand, recommend, recommendations_frame
from elastica.simulate import SimulationConfig, simulate

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pandas as pd

EXIT_OK = 0
EXIT_FAILURE = 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    try:
        return handler(args)
    except (DataError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="elastica",
        description="Estimate causal price elasticity with double ML and recommend prices.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(required=True, metavar="COMMAND")

    sim = commands.add_parser("simulate", help="write synthetic sales with known elasticities")
    sim.add_argument("-o", "--output", type=Path, required=True, help="output .csv")
    sim.add_argument("--products", type=int, default=4)
    sim.add_argument("--days", type=int, default=730)
    sim.add_argument("--seed", type=int, default=7)
    sim.set_defaults(handler=_simulate)

    est = commands.add_parser("estimate", help="estimate the price elasticity of each product")
    _add_data_arguments(est)
    est.add_argument("-o", "--output", type=Path, help="also write results to .csv")
    est.set_defaults(handler=_estimate)

    price = commands.add_parser("price", help="recommend a price for each product")
    _add_data_arguments(price)
    price.add_argument("--unit-cost", type=float, help="unit cost for every product")
    price.add_argument(
        "--band",
        default="0.8,1.2",
        help="lowest and highest price as multiples of the current one (default 0.8,1.2)",
    )
    price.add_argument("-o", "--output", type=Path, help="also write results to .csv")
    price.set_defaults(handler=_price)
    return parser


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("data", type=Path, help="transactions .csv or .parquet")
    columns = parser.add_argument_group("columns")
    columns.add_argument("--date", default="date")
    columns.add_argument("--price", default="price")
    columns.add_argument("--quantity", default="quantity")
    columns.add_argument("--product", default="product", help="use '' for a single product")
    columns.add_argument("--cost", help="column with the unit cost")
    columns.add_argument("--controls", default="", help="comma-separated control columns")
    model = parser.add_argument_group("model")
    model.add_argument(
        "--frequency", choices=[f.value for f in Frequency], default=Frequency.DAY.value
    )
    model.add_argument("--pooled", action="store_true", help="one estimate across products")
    model.add_argument("--folds", choices=[f.value for f in Folds], default=Folds.RANDOM.value)
    model.add_argument("--n-folds", type=int, default=5)
    model.add_argument("--repeats", type=int, default=3)
    model.add_argument("--confidence", type=float, default=0.95)
    model.add_argument("--seed", type=int, default=0)


def _simulate(args: argparse.Namespace) -> int:
    config = SimulationConfig(n_products=args.products, n_days=args.days, seed=args.seed)
    sim = simulate(config)
    sim.transactions.to_csv(args.output, index=False)
    print(f"wrote {len(sim.transactions)} rows to {args.output}", file=sys.stderr)
    print("true elasticities: " + ", ".join(f"{k} {v}" for k, v in sim.true_elasticity.items()))
    return EXIT_OK


def _run_estimates(args: argparse.Namespace) -> list[ProductEstimate]:
    schema = Schema(
        date=args.date,
        price=args.price,
        quantity=args.quantity,
        product=args.product or None,
        cost=args.cost,
        controls=tuple(c.strip() for c in args.controls.split(",") if c.strip()),
    )
    panel = prepare(read_transactions(args.data), schema, Frequency(args.frequency))
    if panel.dropped:
        print(f"dropped {panel.dropped} period(s) with zero sales or price", file=sys.stderr)
    options = EstimateOptions(
        folds=Folds(args.folds),
        n_folds=args.n_folds,
        n_repeats=args.repeats,
        confidence=args.confidence,
        seed=args.seed,
    )
    return estimate(panel, options, pooled=args.pooled)


def _estimate(args: argparse.Namespace) -> int:
    estimates = _run_estimates(args)
    frame = estimates_frame(estimates)
    level = round(args.confidence * 100)
    print(
        render(
            frame,
            {
                "product": "Product",
                "elasticity": "Elasticity",
                "ci_low": f"{level}% CI low",
                "ci_high": f"{level}% CI high",
                "naive_elasticity": "Naive",
                "n_obs": "Periods",
            },
        )
    )
    _print_warnings(frame)
    _write(frame, args.output)
    return EXIT_OK if any(e.result for e in estimates) else EXIT_FAILURE


def _price(args: argparse.Namespace) -> int:
    lower, upper = (float(v) for v in args.band.split(","))
    band = PriceBand(lower, upper)
    recommendations = [
        recommend(e, unit_cost=args.unit_cost, band=band) for e in _run_estimates(args)
    ]
    frame = recommendations_frame(recommendations)
    print(
        render(
            frame,
            {
                "product": "Product",
                "elasticity": "Elasticity",
                "current_price": "Current",
                "recommended_price": "Recommended",
                "price_low": "Range low",
                "price_high": "Range high",
                "profit_change": "Profit",
                "revenue_change": "Revenue",
            },
        )
    )
    _print_warnings(frame)
    _write(frame, args.output)
    return EXIT_OK


PERCENT_COLUMNS = {"profit_change", "revenue_change", "quantity_change"}


def render(frame: pd.DataFrame, columns: dict[str, str]) -> str:
    rows = [list(columns.values())]
    for record in frame.to_dict("records"):
        rows.append([_cell(name, record[name]) for name in columns])
    widths = [max(len(row[i]) for row in rows) for i in range(len(columns))]
    lines = []
    for index, row in enumerate(rows):
        cells = [row[0].ljust(widths[0])] + [
            cell.rjust(width) for cell, width in zip(row[1:], widths[1:], strict=True)
        ]
        lines.append("  ".join(cells))
        if index == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)


def _cell(name: str, value: object) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    if isinstance(value, float):
        return f"{value:+.1%}" if name in PERCENT_COLUMNS else f"{value:,.2f}"
    return str(value)


def _print_warnings(frame: pd.DataFrame) -> None:
    sys.stdout.flush()
    for record in frame.to_dict("records"):
        if record["warnings"]:
            print(f"\n{record['product']}: {record['warnings']}", file=sys.stderr)


def _write(frame: pd.DataFrame, output: Path | None) -> None:
    if output is not None:
        frame.to_csv(output, index=False)
        print(f"wrote {output}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
