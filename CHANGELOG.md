# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-10-26

First public release.

### Added

- Double machine learning estimator for the partially linear model with cross-fitting (random or time-blocked folds), repeated splits, orthogonal-score standard errors, naive and linear-OLS benchmarks and identification diagnostics.
- Transaction loading with column mapping, validation, aggregation to day / week / month and calendar features.
- Per-product and pooled (fixed effects) estimation.
- Price recommendations: Lerner rule, guard band around the current price, price range from the confidence interval, expected quantity / revenue / profit change, plain-language warnings.
- Synthetic data generator with known elasticities and non-linear confounding.
- `elastica` CLI with `simulate`, `estimate` and `price`.
- Reproducible benchmark (`examples/benchmark.py`).

[1.0.0]: https://github.com/DiegoTepichin/elastica/releases/tag/v1.0.0
