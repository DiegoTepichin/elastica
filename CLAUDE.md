@../briefs/cafe-elasticity.md

# cafe-elasticity

## What it is

A library that takes a CSV of transactions, estimates price elasticity with causal methods (double machine learning) and suggests a price. The project name is provisional.

## Who it is for

Data scientists and pricing analysts in retail and e-commerce.

## Scope (v0.1)

To be defined in `docs/PLAN.md` during the planning session.

Anything else is out of scope until v0.1 ships.

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

None yet. Add each command here once it exists and has been run.

## Structure

Not defined yet. Decided in `docs/PLAN.md`.
