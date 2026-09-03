# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

One entry per milestone. An entry says what changed and, where the choice was
not obvious, why — a changelog that only lists filenames is a `git log` with
extra steps.

## [Unreleased]

## [0.3.0] — 2026-09-03

Milestone 3: storage and validation. The canonical table becomes queryable
through an interface, and the assertions that lived in a test file become a
report the pipeline runs on every ingest.

### Added

- `src/storage/` — a `MatchStore` protocol and a DuckDB implementation over the
  Parquet the ingest pipeline writes. Views rather than tables, so the file
  stays the single source of truth; filters (`competitions`, `since`, `until`,
  `columns`) are pushed into the query, which makes a point-in-time read the
  interface's own operation rather than something every caller reimplements.
- `src/validation/` — 23 checks in four groups (schema, integrity, referential,
  distribution) producing a report with three outcomes and two severities. Run
  by `run_ingest` on the frame it just wrote, and by
  `scripts/validate_data.py`, which exits non-zero on a blocking failure.
- `PRE_MATCH_COLUMNS` / `POST_MATCH_COLUMNS` / `BENCHMARK_COLUMNS` on the
  canonical schema, and a check that every column is classified. The leakage
  defence has to be temporal, not columnar: `home_shots` is kept because a
  team's shots in *earlier* matches are a legitimate feature, so it cannot be
  enforced by leaving data out.
- `docs/DATASET_CARD.md`, generated from the table on every validation run,
  with per-competition coverage and the checksum of the exact file it
  describes.
- `make validate`, `make validate-strict`.
- A CI invariant: the canonical table is read through `src/storage`, nowhere
  else.

### Changed

- `IngestReport` carries a `validation` report. It reports rather than gates —
  a table you can inspect is more useful than one the pipeline refused to save.
- `tests/integration/test_real_provider_files.py` no longer restates the data
  assertions. They live in `src/validation` and the integration suite runs
  them, rather than being the same rule in two places.

### Found

- **One row in 305,499 is filed under the wrong season** — an Argentinian match
  played 2015-01-29 and labelled 2013-14 in the provider's own file, 213 days
  outside the widest defensible window. Reported as a warning, left in place.
  The first thing the new gate caught, on the data that already existed.
- **`_goals_are_plausible` crashed on an empty table.** `min()` over an empty
  nullable column returns `pd.NA`, and `pd.NA < 0` raises rather than being
  falsy — so a validation suite would have died at the moment a report was most
  useful. Caught by the empty-table test, not by review.
- **DuckDB's dtypes depend on the rows selected**: a NumPy `int16` for a column
  with no nulls and a nullable `Int16` for one with them. Every store read
  re-asserts the canonical schema, which is also what makes a store read equal
  a `pd.read_parquet` of the same file.

### Deferred, with a reason

- **PostgreSQL and `docker-compose.yml`.** Scoped for this milestone, and
  moved to Milestone 11. There is no application data and no prediction to
  serve yet, so a Postgres store today would be an unused implementation of an
  interface with one caller, plus a compose file nobody runs — the dead code
  the project's own standing rule says to avoid. The seam that makes it cheap
  (`MatchStore`) is delivered.
- **MLflow.** Arrives with the model zoo in Milestone 8, when there is a run to
  track.
- **pandera.** The checks here are mostly semantic — home advantage, a book's
  overround, a season's date window — not schema, and pandera's schema model
  would be a second copy of `CANONICAL_SCHEMA` to keep in step.

## [0.2.1] — 2026-09-03

Incremental, idempotent re-runs. Verifying the claim that a re-run is cheap
found that it was not: settled files were trusted forever so a provider
correction was permanently invisible, live files re-downloaded on age even when
unchanged, unavailable seasons were re-probed every run, and nothing recorded
the checksum of a single input file.

### Added

- HTTP conditional requests (`If-None-Match` / `If-Modified-Since`). The
  provider answers both with 304 and no body, so change detection is
  authoritative instead of guessed. `HttpClient.download` now returns a
  `DownloadResult` reporting whether anything was transferred.
- `src/ingestion/cache.py` — per-URL `ETag`, `Last-Modified`, `sha256` and
  last-checked time, plus recorded misses with a 7-day TTL so unpublished
  seasons are not re-probed every run. A corrupt or version-mismatched cache is
  discarded rather than raised on.
- Raw-file manifest (`data/raw/manifest.json`) checksumming all 732 cached
  provider files — provenance of the inputs, alongside the existing manifest
  for the output.
- `--revalidate` re-checks finished seasons to pick up a provider correction;
  `--forget-misses` retries seasons previously recorded as unpublished.
  `make refresh` and `make revalidate`.
- 36 tests covering incremental re-runs, conditional requests and the cache,
  plus integration tests asserting the real manifest still verifies.

### Changed

- A secondary-feed country file is now revalidated **once per run** rather than
  once per season. Ingesting Brazil's fifteen seasons asked the server about
  the same file sixteen times.
- `max_age_days` removed. Age was the wrong question in both directions.

### Measured

Two full 39-competition ingests back to back: the second transferred **0 files
and 0.00 MB**, re-probed **0** of the previous run's misses, and produced a
**byte-identical** table — `sha256 8d489ed3…`, unchanged from before this work,
so the fetching got smarter and the data did not move.

## [0.2.0] — 2026-09-03

Milestone 2: ingestion.

### Added

- Provider-adapter layer (`src/ingestion/`). 39 competitions across 27
  countries and two provider file layouts reduce to one canonical 35-column
  schema; a full ingest yields 305,499 matches over 1993-2026. CI enforces that no module outside `src/ingestion` names the
  provider, so adding a source cannot leak into downstream code.
- Competition registry (`configs/leagues.yaml`). Adding a league is a config
  entry; `league_filter` splits country files that mix a league with a cup
  (Argentina) or two tiers (Switzerland).
- Season-label normalisation covering both split (`2024-25`) and calendar
  (`2024`) seasons, read per row — Japan's J1 League switches between them.
- Robust CSV reading: `utf-8-sig`/`cp1252` fallback, BOM handling, blank-row
  and ragged-row repair, and an HTML-served-as-CSV guard.
- Canonical team identity scoped to country (`eng:man-united`), stable across
  promotion and relegation, with a rename detector.
- Checksum manifests (`src/ingestion/manifest.py`) as dataset versioning.
- `scripts/fetch_data.py`, `make data`, `make leagues`, `make test-int`.
- `docs/DATA_SOURCES.md` — thirteen measured provider quirks, each naming the
  file it was found in.
- 16 integration tests asserting football-shaped invariants on real data
  (home-win rate, half-time ≤ full-time, shots on target ≤ shots, odds implying
  an overround above 100%). All skip without downloaded data.

### Changed

- Ragged-row handling now warns instead of raising. The original rule — surplus
  non-empty means a shifted row — was falsified by Italian Serie B 2003/04,
  whose 42-column header has eight blank names and whose 49-field rows are
  correctly aligned. Raising also aborted the entire run on one bad file out of
  roughly seven hundred.
- `PathsConfig.sample_dir` removed. No data slice is committed: the provider
  publishes no redistribution licence, and the unit suite is already offline.
- Every config section is now optional, since every field within one has a
  default.

### Notes

- No fuzzy team-name matching, and that was a measurement: 51 distinct team
  strings across 34 Premier League seasons, with the two most similar distinct
  pairs at exactly 0.800 similarity. Any threshold that catches a rename merges
  Sheffield United with Sheffield Wednesday.

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

[Unreleased]: https://github.com/Vanshcloud/Match-Outcome-Predictor/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/Vanshcloud/Match-Outcome-Predictor/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Vanshcloud/Match-Outcome-Predictor/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Vanshcloud/Match-Outcome-Predictor/releases/tag/v0.1.0
