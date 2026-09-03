# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

One entry per milestone. An entry says what changed and, where the choice was
not obvious, why — a changelog that only lists filenames is a `git log` with
extra steps.

## [Unreleased]

## [0.6.0] — unreleased

Milestone 6: the leakage suite. Two more probes, and a change of principle —
the suite no longer takes a list of what to check.

### Added

- `src/validation/leakage.py`. It **walks** `src.ratings` and
  `src.feature_engineering` and finds every class satisfying the producer
  contract, constructed with the defaults the pipelines use. Both earlier
  milestones probe the producers on their own list, and a list is a thing you
  can forget to add to: a builder wired into a pipeline but omitted from its
  probe call would have shipped unverified, with nothing in the output saying
  so. `check_defaults_are_complete()` closes the other half — a discovered
  producer in no pipeline's defaults is a column the model layer expects and
  will not get.
- **Split boundary**, the third probe. No training row may be dated at or
  after any evaluation row, and no match may appear in both halves. Ties fail:
  a full Saturday programme is one round, and a model trained on the 3pm
  results is not entitled to predict the 5.30 kick-off. Milestone 7 owns the
  splits; the probe is here and tested, waiting for them.
- **Observed reads**, the fourth. Rewrite one input column, recompute, and
  whatever moved read it. This is the measured counterpart to the feature
  registry's hand-written `reads` — the one part of that registry that can be
  wrong without anything noticing, since a typo there reclassifies a leaking
  feature as safe. The suite asserts the declaration *covers* what was
  measured; over-declaring is safe, under-declaring is not.
- `benchmark_leaks()`: any derived column that moves when a bookmaker's price
  is rewritten fails the run. The rule was written on `ODDS_COLUMNS` in
  Milestone 2 and until now was only written.
- `tests/unit/test_leakage_suite.py` — both temporal probes over every
  discovered producer, on every test run. A unit test rather than a shell step
  on purpose: CI runs the suite, so a producer added on a branch is probed by
  the same command its author already runs locally. CI names it as its own
  step so a leak appears in the job list rather than as one dot among five
  hundred.
- `scripts/audit_columns.py`, `make audit`, and **[docs/LEAKAGE.md](docs/LEAKAGE.md)**:
  every derived column the model layer will see, traced back to the canonical
  columns it reads.

### Measured

Thirty derived columns traced over the most recent 1,200 matches of ENG_1 and
ESP_1. Every probe held; nothing reads `result`, `ht_*`, `referee` or an odds
column. Four things the trace says that reading the code does not:

| Finding | Why it matters |
|---|---|
| Elo reads `season`, never `date` | Its ratings are invariant to *when* matches were played and sensitive only to order — correct for an online update, and worth knowing before anyone adds time decay to it. |
| Dixon-Coles reads `competition_id`; Elo reads no partition column at all | Elo's per-country pooling is already inside `home_team_id`, which is country-scoped so a promoted club keeps one id. The partition is in the identifier, not in a `groupby`. |
| `home_venue_points_5` does not read `away_team_id` | Venue form is a team's record at its own end and the opponent is irrelevant to it. This is the asymmetry a reshape is most likely to get wrong, and it shows up as an absence rather than as a number to check by eye. |
| `elo_*_played` reads no result | The same conclusion the registry reaches by declaration for `*_matches_played`, reached here by measurement. |

The trace is a **lower bound**, and both ways it falls short are properties of
the sample rather than of the code. A column that is entirely null cannot be
perturbed: run the audit over ENG_1's first 2,000 matches and it reports that
no feature reads shots — correctly, for that slice, since the provider carried
none before 2000/01. A column that does not vary cannot be perturbed either, so
a single-competition sample cannot show that Dixon-Coles partitions on
`competition_id`. Hence the most recent matches, from two competitions in two
countries.

### Changed

- `--limit` on the audit takes the most **recent** N matches per competition
  rather than the first. The first N are the ones with no shot data, and an
  audit over them reports absences that belong to 1993 rather than to the code.

## [0.5.0] — unreleased

Milestone 5: feature engineering. Twenty features, and the correctness bug that
building them uncovered.

### Added

- `src/feature_engineering/` — 20 features in three groups. Form, venue form,
  goals and shots for and against, rest days, fortnight congestion, and
  head-to-head record.
- `windows.py`, the causal primitive: every window ends at the last row
  **strictly earlier by date**. `shift(1).rolling(k)` counts rows, so two
  matches on the same date let the earlier one — earlier only by an arbitrary
  tiebreak — inform the later.
- `registry.py`, where each feature declares the canonical columns it reads.
  Whether it *can* leak is derived from that rather than declared separately,
  so there is no field to set wrongly. Seven of the twenty read nothing but the
  fixture list.
- `src/validation/features.py` — 10 checks, including a table-level leakage
  gate: every window feature has a companion count, and a value where the count
  is zero came from somewhere it should not have. It runs over every row
  written, where the probes see one sampled competition.
- `src/pipelines/features.py`, `scripts/build_features.py`, `make features`
  (~10 seconds) and `make feature-list`.
- A CI invariant: `src/feature_engineering` may import the canonical schema and
  `src/utils`, nothing else. A builder that reached for the store could pass
  every probe while reading the future, because the probes work by handing it
  truncated frames.

### Changed

- **`src/pipelines/derived.py` extracted.** The ratings and feature pipelines
  had the same shape — run producers, probe, check, persist, report — and
  written twice the two would drift, with the drifting half always being the
  probe nobody looked at again. The ratings pipeline moved onto it with its
  tests unchanged.
- Feature checks are scoped to what a build claims to have produced, so
  `--builder head_to_head` is not reported as twelve null columns. A warning
  that fires on a legitimate command is one people learn to ignore.

### Measured

Home win rate by the gap between the two sides' five-match form:

| Form gap | Matches | Home | Draw | Away |
|---|---:|---:|---:|---:|
| below −1.2 | 38,186 | 31.7% | 27.2% | 41.1% |
| −0.2 to 0.2 | 61,169 | 45.1% | 27.5% | 27.3% |
| above 1.2 | 26,201 | 61.1% | 22.2% | 16.7% |

A thirty-point spread from one feature — and the draw rate peaks between evenly
matched sides and falls at both extremes, which football says should happen and
the feature was not built to produce.

Venue form is worth ±0.275 points per game, symmetric between the two sides to
three decimals: a home team takes 1.618 at home against 1.343 overall, an away
team 1.119 away against 1.392.

**Rest days carry no marginal signal**: 43.7% to 45.6% home wins across the
range, and not even monotone. Kept for the interaction rather than the
marginal; Milestone 8's ablation is where it earns its place or is dropped.

### Fixed

- **Five of the provider's early files are copies of `SP1.csv` served under
  another name**, and the adapter believed the URL. `1993-94/P1.csv`,
  `1993-94/SC1.csv` and `SP2.csv` for 1993-94, 1994-95 and 1995-96 all carry
  Spanish La Liga rows — with `Div` inside each file saying `SP1`. The result
  was 380 fabricated Portuguese matches and 380 fabricated Scottish ones,
  Spanish clubs wearing `por:` and `sco:` team ids, plus 1,222 La Liga
  fixtures duplicated into Spain's second division.

  Every copy was internally consistent — same teams, same date, same score —
  so no per-row check could see it, and `match_id` hashes the competition, so
  deduplication kept both. The adapter now trusts each file's own `Div` column
  and drops rows that name a different division; a new validation check,
  **"no fixture appears in two competitions"**, is what says it worked.

  Found by an assumption check written for the feature layer: rolling form
  needs a team never to play twice on one date, and 2,444 team-days said
  otherwise.

### Changed

- The ingest is **303,517 matches and 1,323 teams**, down from 305,499 and
  1,363. Portugal's first league season now starts 1994-08-20 and Scotland's
  second tier 1994-08-13, rather than both starting in Spain in September 1993.
- Every measured figure moved in the fourth decimal. Elo's error over the
  corrected table is 0.16177 against 0.16169 before; the Premier League
  Dixon-Coles numbers are unchanged, because England was never affected.

## [0.4.0] — 2026-09-03

Milestone 4: ratings. Two of them, and the machinery that proves neither can
see the match it is rating.

### Added

- `src/ratings/elo.py` — one Elo pool per country, online, with home advantage,
  a margin-of-victory multiplier, autocorrelation damping and season
  carry-over. Online is the point: the rating carried into match *n* is a
  function of matches 1..*n*-1 by construction.
- `src/ratings/dixon_coles.py` — bivariate Poisson with the low-score
  correction and exponential time decay, refitted per competition on a rolling
  window that ends **strictly before** the match that triggered the refit. It
  produces a real H/D/A distribution, so it is the first thing here that can be
  scored against a bookmaker.
- `src/validation/temporal.py` — prefix invariance and outcome independence,
  generic over `Callable[[DataFrame], DataFrame]`. Scoped for Milestone 6 and
  written here, because shipping two rating models with their central property
  unverified for two milestones is not a trade worth making.
- `src/validation/ratings.py` — 9 arithmetic and coverage checks on the output.
- `src/pipelines/ratings.py` and `scripts/build_ratings.py`. The build runs the
  probes against a sample competition and exits non-zero when one fails.
- `make ratings` (~10 min) and `make ratings-elo` (seconds).
- A CI invariant: `src/ratings` depends only on `src.ratings` and `src.utils`.
  A model that reached for the store could pass every probe while reading the
  future, because the probes work by handing it truncated frames.

### Measured

Elo, over all 305,499 matches, every prediction from prior matches only —
mean squared error of the expected score:

| | MSE |
|---|---|
| Plain Elo | 0.16882 |
| + home advantage | 0.16217 |
| + margin of victory | 0.16208 |
| + autocorrelation damping | 0.16173 |
| + season carry-over (shipped) | **0.16169** |

Dixon-Coles on the English Premier League. All three computed over exactly the
same 8,818 matches — the ones the model priced *and* the provider carries a
closing line for:

| | log loss | RPS |
|---|---|---|
| Class prior | 1.0636 | 0.2285 |
| **Dixon-Coles** | **0.9774** | **0.1988** |
| Bookmaker closing odds, overround removed | 0.9619 | 0.1941 |

Which is where the README says a public-data model should land, from a rating
rather than from the model zoo. Over all 12,052 matches it could price,
including three decades before the odds series starts, log loss is 0.9888.

The full build, measured: 305,499 matches rated in about ten minutes,
Dixon-Coles pricing 92.1% of them and Elo all of them, all four causality
probes holding, and 9 of 9 checks passing with none skipped.

### Changed

- Elo's `season_carry` ships at 0.97, not the football-Elo convention of 0.75.
  Three decades of results say a club's strength persists across a summer far
  more than the convention assumes.
- Dixon-Coles ships with a 347-day decay half-life and a three-season window,
  both longer than the literature's, and refits every 60 days rather than 30 —
  which measured no worse and halves the runtime.

### Removed before shipping

- **Per-competition Elo fitting**, which the plan called for. Built, measured,
  removed: it made the ratings worse (0.16300 against 0.16246), because a
  calibration window is the coldest part of the history and three free
  parameters chase warm-up noise. `--fit-until` re-derives the constants as a
  maintenance step.

### Fixed

- `DuckDBStore.open_matches` attached the matches view, failed on a missing
  ratings file, and left a file-backed catalog holding half of what was asked
  for — which the next open would have reported as success. Every source path
  is checked before the catalog is touched.

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
