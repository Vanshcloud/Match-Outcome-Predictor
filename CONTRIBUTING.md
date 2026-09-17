# Contributing

Thanks for taking an interest. This is a personal project maintained by one
person, so reviews may take a few days.

## What is welcome

- **Bug reports**, especially a wrong number on the dashboard, a page that
  crashes, or a claim in the documentation the code does not support.
- **Leakage or evaluation concerns.** If a feature, split or metric could be
  reading information that is not available before kick-off, that matters more
  than anything else here. Please include the column or function.
- **Small, focused fixes** to code, tests or documentation.

Changes to the model, the features or the evaluation protocol need a
measurement to go with them: the walk-forward log loss and RPS before and
after, on the same folds. A change that improves a number without that evidence
will not be merged.

## Reporting a bug

Open an issue using the bug report template. Include what you ran, what you
expected, and what happened, with the full error text. For anything
security-sensitive, follow [SECURITY.md](SECURITY.md) instead of opening a
public issue.

## Setting up

Python 3.13 is required. On macOS, LightGBM and XGBoost also need
`brew install libomp`.

```bash
python3.13 -m venv .venv
make install-dev   # dev dependencies, without the maintainer's git hooks
make test          # unit suite: offline, no data needed
```

`make setup` does the same and also installs `scripts/hooks`, whose
`commit-msg` hook only accepts the maintainer as author. Use `make install-dev`
unless you are the maintainer.

The unit suite needs no data. To run the pipelines or the dashboard against real
data, see the Quick start in the [README](README.md).

## Checks a change must pass

CI runs these on every pull request, and they can all be run locally:

| Check | Command |
|---|---|
| Unit tests with 100% statement coverage of `src`, `api` and `dashboard` | `make test-cov` |
| Lint (ruff) and formatting (black) | `make lint`, `make format-check` |
| Strict type checking (mypy) | `make typecheck` |
| Architectural boundaries | `make invariants` |

`make quality` runs lint, format check, types and invariants together, and
`make format` applies black and ruff's safe fixes.

A few conventions the checks enforce or reviews will ask for:

- Every behaviour change comes with a test. A bug fix comes with a test that
  fails without it.
- Deprecation warnings are errors in the test suite.
- `api` may import `src`; nothing in `src` imports `api`; the dashboard talks
  to the API over HTTP and never imports it.
- No secrets in code, config, tests or screenshots. Configuration that is a
  credential comes from the environment (see [.env.example](.env.example)).

## Pull requests

1. Open an issue first for anything larger than a small fix, so the approach
   can be agreed before you spend time on it.
2. Branch from `main` and keep the change focused on one thing.
3. Run `make quality` and `make test-cov` before pushing.
4. Describe what changed, why, and how you verified it.

**About commit authorship.** This repository's history is kept to a single
author, and CI's *Authorship* job fails on commits by anyone else. That job
will be red on an outside pull request. If a change is accepted, the maintainer
will apply it and credit you by name in the pull request and the
[CHANGELOG](CHANGELOG.md).

## Code of conduct

Participation is covered by the [Code of Conduct](CODE_OF_CONDUCT.md).
