# AGENTS.md

Operational rules for any coding agent (Claude Code, Gemini CLI, Codex, etc.) working on `elastica`. Project context lives in `CLAUDE.md`.

## Before changing code

- Read `CLAUDE.md` and, if present, `docs/PLAN.md`. Stay within the scope in `docs/PLAN.md`.
- If a request falls outside that scope, say so and propose filing it as an issue instead.

## Workflow

1. Create a branch from `main`: `feat/<topic>`, `fix/<topic>`, `chore/<topic>` or `docs/<topic>`.
2. Make small, atomic commits using Conventional Commits in English.
3. Before pushing, run `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy .` and `uv run pytest`. All must pass.
4. Open a pull request. Merge only after CI is green, locally with `git merge --no-ff`, then push `main` and delete the branch. Never merge from the web UI.

## Never without explicit approval from the maintainer

- Force-push or rewrite published history.
- Change repository visibility, delete repositories, or delete remote branches you did not create.
- Publish packages (PyPI, npm) or create releases.
- Add dependencies that are not needed by the current milestone.

## Always

- Keep secrets out of git. Use environment variables and keep `.env.example` current.
- Add or update tests with every behavior change.
- Run a command before documenting it. Document only what exists.
- Do not add AI attribution (for example `Co-Authored-By` trailers) to commits or pull requests.
