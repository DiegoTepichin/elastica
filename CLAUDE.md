@../briefs/elastica.md

# elastica

## What it is

A library that takes a CSV of transactions, estimates price elasticity with causal methods (double machine learning) and suggests a price. It is written from scratch on public methods.

## Who it is for

Data scientists and pricing analysts in retail and e-commerce.

## Scope

v1.0 scope and out-of-scope list live in `docs/PLAN.md`. Ideas beyond it go to GitHub issues. No code from CAFE, ever.

## Code standards

- Python 3.11+ (CI matrix: 3.11, 3.12, 3.13, 3.14). Dependencies and environments managed with `uv`.
- Strict typing: `mypy --strict` clean, no untyped public functions, no `Any` without a reason.
- Lint and format with `ruff`.
- Tests with `pytest`. No network calls in tests: use recorded or fixture data.
- Small functions, clear names, no dead code, no comments that restate the code.
- Configuration and secrets only through environment variables, documented in `.env.example`.
- Conventional Commits in English, one logical change per commit.
- README and docs describe only what exists and has been run.

## Commands

- `uv sync`: install the project and dev dependencies (`--all-groups` adds matplotlib for the benchmark).
- `uv run ruff check .` / `uv run ruff format --check .` / `uv run mypy .` / `uv run pytest`: the CI checks.
- `uv run elastica simulate|estimate|price ...`: the CLI (see README).
- `uv run --group docs python examples/benchmark.py [n]`: regenerate `docs/benchmark.svg` and the README table.
- `uv build`: build the sdist and wheel.

## Structure

- `src/elastica/`: `simulate`, `data`, `dml`, `estimate`, `pricing`, `cli`.
- `tests/`: one module per source module; statistical recovery tests use fixed seeds.
- `examples/benchmark.py`: the README benchmark.
- `docs/`: `PLAN.md` and `benchmark.svg`.
