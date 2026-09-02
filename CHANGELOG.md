# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

One entry per milestone. An entry says what changed and, where the choice was
not obvious, why — a changelog that only lists filenames is a `git log` with
extra steps.

## [Unreleased]

## [0.1.0] — 2026-09-03

Milestone 1: repository foundation.

### Added

- Typed, frozen configuration (`src/utils/config.py`): YAML defaults overlaid
  by an explicit map of environment variables, validated by Pydantic v2.
  Unknown keys are an error rather than being silently ignored.
- Project-root-anchored path resolution (`src/utils/paths.py`), so no module
  computes a location relative to its own file.
- Idempotent logging setup (`src/utils/logging.py`) writing to stderr.
- A polite retrying HTTP client (`src/utils/http.py`) with atomic downloads:
  a truncated file is deleted rather than left to be parsed as a short season.
- Quality gates: ruff, black, mypy (strict), pytest. 71 tests, 100% coverage
  of `src`.
- `commit-msg` authorship hook, installed via version-controlled
  `core.hooksPath` so it survives a reclone.
- GitHub Actions CI running the same gates a developer runs locally.

### Notes

- `requirements.txt` grows one milestone at a time. A pinned dependency that
  nothing imports is a supply-chain surface with no upside.
- Pydantic does not run field validators on defaults unless
  `validate_default=True`. Without it, `data_dir` resolved to an absolute path
  when named in the config and a relative one when omitted — every downstream
  path would then have depended on the process's working directory. Fixed at
  the base model so no future section can reintroduce it.

[Unreleased]: https://github.com/Vanshcloud/Match-Outcome-Predictor/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Vanshcloud/Match-Outcome-Predictor/releases/tag/v0.1.0
