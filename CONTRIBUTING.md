# Contributing

Thanks for your interest. Issues and pull requests are welcome, in English or Spanish.

## Before you start

- For anything bigger than a small fix, open an issue first so we can agree on the approach.
- Check the open issues to avoid duplicate work.

## Reporting a wrong or surprising estimate

Include the command you ran, the output and, if you can, a small anonymized or simulated dataset that reproduces it (`elastica simulate` is a good starting point). Never share real sales or cost data you are not allowed to publish.

## Development setup

```bash
uv sync
```

## Checks

All of these must pass before a pull request is merged (CI runs them too):

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest
```

## Pull requests

- One logical change per pull request.
- Add or update tests for every behavior change. Statistical tests must use fixed seeds and simulated data.
- Update both READMEs if you change user-facing behavior.
- Use [Conventional Commits](https://www.conventionalcommits.org/) in English, for example `feat: add cross-price elasticity` or `fix: handle constant controls`.

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
