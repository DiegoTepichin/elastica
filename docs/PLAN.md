# elastica: plan to v1.0

## Goal

From a CSV of transactions, estimate the causal price elasticity of demand per product with double machine learning (DML), report how much a naive regression would have been off, and turn the estimate into a price recommendation with explicit assumptions and uncertainty.

The value is the pricing workflow around the estimator (data preparation, identification diagnostics, guarded price optimization), not the estimator alone.

## Method

Partially linear model on log-log data, per product:

```
log q = θ · log p + g(X) + ε        θ = price elasticity
log p =         m(X) + v
```

1. Cross-fit `E[log q | X]` and `E[log p | X]` with scikit-learn learners (K folds, optionally in contiguous time blocks, which limits leakage between neighbouring periods).
2. Regress outcome residuals on price residuals: `θ̂ = Σ ṽ·ũ / Σ ṽ²`.
3. Heteroskedasticity-robust standard error from the DML orthogonal score; normal confidence interval.
4. Repeat cross-fitting with different splits and take the median (Chernozhukov et al., 2018).

`X` holds user controls (promotions, stores, holidays…) plus calendar features derived from the date (trend, week of year, month, day of week).

## Pricing

Constant-elasticity demand `q = A · p^θ` around the observed price. With unit cost `c`, the profit-maximizing price is `p* = c · θ / (1 + θ)` when `θ < -1`. The recommendation is clamped to a user-set band around the current price (default ±20 %), and also computed at both ends of the elasticity's confidence interval. Inelastic estimates (`θ ≥ -1`) return the band edge with a warning: the model is not valid far from observed prices.

## Architecture

| Module | Responsibility |
|---|---|
| `data` | Column mapping, validation, aggregation of transactions to product × period, calendar features. |
| `dml` | Cross-fitted partially linear estimator, standard errors, repetitions, naive OLS benchmark, diagnostics. |
| `estimate` | Per-product estimation over a prepared panel; result tables. |
| `pricing` | Optimal price, guardrails, demand / revenue / profit curves. |
| `simulate` | Synthetic transactions with a known elasticity and confounding, for tests and demos. |
| `cli` | `elastica estimate`, `elastica price`, `elastica simulate`. |

## Decisions

- **Dependencies:** numpy, pandas, scikit-learn. The normal quantile comes from `statistics.NormalDist`.
- **Default learner:** `HistGradientBoostingRegressor` (handles non-linearities and categorical-like controls, fast). Any scikit-learn regressor can be passed.
- **Price signal check:** if the residual price variance is tiny relative to raw price variance, the estimate is flagged as weakly identified.
- **No CAFE code.** Everything is written from public literature.

## Scope v1.0

1. Data loading and aggregation with a documented schema.
2. DML estimator with cross-fitting, time-blocked folds, repetitions, robust CI and naive benchmark.
3. Per-product estimates with diagnostics and minimum-data checks.
4. Price recommendation with cost, guardrails and uncertainty; demand / profit curves.
5. Synthetic data generator with ground truth.
6. CLI; statistical tests that recover a known elasticity; CI.

## Out of scope

Real-time pricing, e-commerce integrations, cross-price elasticities, a web UI, heterogeneous effects beyond per-product estimates.

## Milestones

| # | Branch | Content |
|---|---|---|
| M0 | `chore/tooling` | pyproject, CI, plan. |
| M1 | `feat/simulate` | Synthetic data with known elasticity. |
| M2 | `feat/data` | Loading, validation, aggregation, calendar features. |
| M3 | `feat/dml` | Estimator, inference, diagnostics. |
| M4 | `feat/estimate` | Per-product estimation. |
| M5 | `feat/pricing` | Optimal price and curves. |
| M6 | `feat/cli` | CLI. |
| M7 | `docs/v1` | README (EN / ES) with a real benchmark chart, example notebook-free walkthrough, release 1.0.0. |

## Test strategy

- Unit tests per module.
- Statistical recovery tests on simulated data with fixed seeds: DML within tolerance of the true elasticity, naive OLS measurably biased, CI coverage sanity check.
- CLI tests through `main(argv)`.

## Launch checklist

- [x] Checks green on main (run locally; Actions paused by budget on private repos).
- [x] README with a benchmark chart produced by a committed script.
- [ ] Wheel installs and runs `elastica simulate | elastica estimate`.
- [ ] Tag `v1.0.0` (PyPI only with the maintainer's OK).
