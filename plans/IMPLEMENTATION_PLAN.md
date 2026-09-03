# Implementation plan

The decisions that shape every milestone, recorded once so they are not
re-litigated later. Milestone sections are filled in as each is delivered.

---

## Approved decisions

| # | Decision | Chosen | Rationale |
|---|---|---|---|
| 1 | Data provider | football-data.co.uk only, behind an adapter interface | Free, keyless, static CSV, no rate limit. Measured after a full ingest: 39 competitions, 303,517 matches, 1993-2026. Verified live before committing to it. Other providers plug in later without downstream change. |
| 2 | Model zoo | Pruned zoo first (M8), then a separate Research Models milestone (M14) | Prove the pipeline end to end with models that reliably win on tabular data, then benchmark TabNet / FT-Transformer / AutoML against the best GBDT under *identical* time-aware validation, with a written verdict on whether the complexity is justified. |
| 3 | Storage & MLOps | DuckDB + Parquet (analytics, feature store), PostgreSQL (application data, prediction serving), MLflow (tracking, registry), Docker Compose (orchestration) | Storage sits behind interfaces so Postgres can be replaced or scaled without touching business logic. Checksum-based dataset versioning now; the layout stays compatible with adding DVC or S3/MinIO later. |
| 4 | Commit policy | `commit-msg` authorship guard, no attribution trailers | Matches the sibling Transfer Value Predictor and predictive-maintenance repositories. Installed via version-controlled `core.hooksPath`, so it survives a reclone. |

### Provider verification (2026-09-03)

Probed before the architecture was written rather than assumed:

- `/mmz4281/{season}/{div}.csv` — all 22 divisions returned HTTP 200 for
  2024/25; seasons 1993/94, 2000/01, 2005/06 and 2010/11 also 200.
  Columns: `Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,
  Referee,HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR` plus ~90 odds columns.
- `/new/{COUNTRY}.csv` — all 16 country files returned HTTP 200.
  Columns: `Country,League,Season,Date,Time,Home,Away,HG,AG,Res` plus closing
  odds. **No shot statistics.**

The two schemas differ in what they carry. That difference is modelled
explicitly (a per-competition capability flag), not papered over: a feature
declares the data it requires and yields null where a competition cannot
supply it.

### Excluded, deliberately

- **Transfermarkt.** ToU §11.1 bans automated access *and* training models on
  the content. A registry slot exists; the adapter will not be written.
- **FBref / Sports Reference.** Cloudflare-blocked; the sibling project's spike
  recorded HTTP 403 on every policy page. Not worth the fragility.
- **FiveThirtyEight SPI.** The feed stopped updating when 538 was wound down;
  the archive is usable history but cannot serve live fixtures.
- **Weather, injuries, lineups, travel, attendance, squad value.** Not
  available at match level across 39 competitions from any free source. Feature
  registry slots exist so any that becomes available is one entry, not a
  refactor.

### Metric decision

Primary: **log loss and Ranked Probability Score.** RPS respects the ordinal
structure of H/D/A — predicting Away when the result was Home is a worse error
than predicting Draw, and log loss alone does not know that. Accuracy,
precision, recall, F1 and ROC-AUC are reported but never selected on.

Benchmark: bookmaker closing odds, converted to probabilities with the
overround removed. Beating the closing line is not the goal; being close to it
on public data, and honest about the gap, is.

---

## Milestone 1 — Foundation ✅

**Delivered.** Repository skeleton, quality gates, and the four utilities every
later milestone depends on.

| Module | Responsibility |
|---|---|
| `src/utils/paths.py` | One anchor. Every path resolves from `PROJECT_ROOT`; nothing uses `../..`. |
| `src/utils/config.py` | YAML defaults → `.env` → real environment, validated into a frozen Pydantic v2 object. Unknown keys are errors. |
| `src/utils/logging.py` | Idempotent stderr handler. Libraries get loggers; entry points configure handlers. |
| `src/utils/http.py` | Retry, backoff, timeout, rate limit, atomic download. Transport only — no caching or parsing, so adapters do not inherit each other's decisions. |

**Verified:** 71 tests, 100% coverage of `src`, ruff/black/mypy clean.

### Two bugs found and fixed during the build

Recorded because both are the silent kind, and both would have been far more
expensive at Milestone 5.

1. **Pydantic does not validate field defaults.** `PathsConfig.data_dir`
   defaults to the relative `Path("data")` and is made absolute by a validator.
   Without `validate_default=True`, a config that *named* the directory got an
   absolute path and one that *omitted* it got a relative one — so every
   downstream path would have depended on the process's working directory,
   which is the exact failure `paths.py` exists to prevent. Fixed on the base
   model so no future config section can reintroduce it.
2. **`HttpClient.__init__` mounts its retry adapter over any pre-mounted one.**
   The test helper mounted a stub first, so it was silently replaced and every
   HTTP test reached the real network — presenting as the suite *hanging* on
   DNS rather than failing. The helper now mounts after construction, and says
   why in a comment.

### Deferred out of Milestone 1, on purpose

- `requirements-lock.txt` — arrives with the container images (M13), when
  there is an image whose resolution must match the developer's.
- mypy per-package `ignore_missing_imports` overrides — added by the milestone
  that introduces the import. Declared early, mypy reports each as an unused
  section, and a gate that prints noise on a clean run is one people skim.
- CONTRIBUTING / SECURITY / ROADMAP / issue templates — M13, with the rest of
  the release furniture.
- `data/sample/` fixtures — M2, when there is real data to slice.

---

## Milestone 2 — Ingestion ✅

**Delivered.** 39 competitions across 27 countries, two provider file layouts,
one canonical 35-column schema. A full ingest produced 303,517 matches and
1,323 teams spanning 1993-2026, with zero unreadable files. No downstream module can tell which feed a
match came from — CI enforces it.

| Module | Responsibility |
|---|---|
| `base.py` | The canonical schema, `Capability`, `MatchProvider` protocol, deterministic `match_id`. |
| `csv_reader.py` | Every file-level quirk in one place: encodings, BOMs, ragged rows, blank rows, HTML-served-as-CSV. |
| `registry.py` | The competition registry and season-label normalisation (split vs calendar). |
| `teams.py` | Canonical team ids, alias seam, rename detection. |
| `football_data.py` | The adapter: URLs, caching, parsing, per-row validation. |
| `manifest.py` | Checksum-based dataset versioning, in place of DVC. |
| `pipelines/ingest.py` | Orchestration only — no logic. |

**Verified:** 213 unit tests, 100% coverage of `src`, ruff/black/mypy clean,
plus 16 integration tests that run against real downloaded data and skip
without it.

### Two decisions that changed after measurement

Recorded because both were wrong first, and both were only caught by running
against all forty competitions rather than a sample.

1. **The ragged-row guard was too strict, and it aborted the run.** It raised
   whenever a truncated field was non-empty, on the theory that surplus data
   meant a shifted row. Italian Serie B 2003/04 falsified it: a 42-column
   header whose last eight names are blank, with rows of 49 fields carrying
   extra unnamed statistics and every named column correctly aligned. Worse,
   raising killed the whole ingest — one bad file out of ~700 discarded every
   competition after it. Now: pad, truncate, warn; the real guard is semantic
   and per-row, and the pipeline survives an unreadable file.
2. **No fuzzy team-name matching.** Measured across 34 Premier League seasons:
   51 distinct team strings, 12,724 matches, byte-identical spellings across
   three decades. The two most similar distinct pairs both score exactly 0.800
   (`Sheffield United`/`Sheffield Weds`, `Barnsley`/`Burnley`), so any useful
   threshold merges real clubs. Identity is exact, with an empty alias table
   for the second provider.

### Changed from the approved plan

- **No `data/sample/`.** The provider publishes no licence granting
  redistribution, and the unit suite is already fully offline on synthetic
  fixtures. Committing a slice would have contradicted the README's own claim.
  `PathsConfig.sample_dir` was removed rather than left unused.

### Incremental re-run verification (post-milestone)

Asked to confirm a re-run is incremental and idempotent, and four of the five
properties did not hold. Recorded because the verification was worth more than
the feature it checked.

| Property | Before | After |
|---|---|---|
| Identical output on repeat | held — Parquet byte-identical | held |
| No duplicate matches | held | held |
| Only **new** files downloaded | HTTP-300 misses re-downloaded every run | recorded with a 7-day TTL |
| Only **modified** files downloaded | undetectable — settled files trusted forever, live files refetched on age | conditional requests, authoritative |
| Historical checksums preserved | no raw manifest existed | all 732 files checksummed |

The design change is that change detection is the *server's* answer, not a
heuristic: the provider serves `ETag` and `Last-Modified` and honours
`If-None-Match` with a 304 carrying no body. A file-age rule is wrong in both
directions — it re-downloads unchanged files once they age, and misses a file
that changed minutes ago.

Measured over two full ingests: the second transferred 0 files and 0.00 MB and
produced the same `sha256` as before the work began.

---

## Milestone 3 — Storage and validation ✅

**Delivered.** The canonical table is queryable through an interface, and the
assertions that lived in a test file are now a report the pipeline runs on
every ingest.

| Module | Responsibility |
|---|---|
| `storage/base.py` | The `MatchStore` protocol. Read-shaped, because nothing but ingestion writes yet. |
| `storage/duckdb_store.py` | DuckDB views over Parquet; point-in-time and per-competition reads pushed into the query. |
| `validation/report.py` | `Check`, three outcomes, two severities, one report. |
| `validation/matches.py` | 23 checks: schema, integrity, referential, distribution. |
| `validation/card.py` | The dataset card, generated from the table. |
| `scripts/validate_data.py` | The gate. Non-zero exit on a blocking failure. |

**Verified:** 378 unit tests, 100% coverage of `src`, ruff/black/mypy clean,
12 integration tests against the real 303,517-row table.

### Three decisions worth recording

1. **Views, not tables.** `CREATE TABLE AS` would have copied 303,517 rows into
   the catalog and then served yesterday's copy after every re-ingest. A view
   keeps the Parquet as the single source of truth, and DuckDB reads it in
   place — so the "database" is a few kilobytes of view definition.
2. **Every read re-asserts the canonical dtypes.** DuckDB returns a NumPy
   `int16` for a column with no nulls and a nullable `Int16` for one with them,
   so the dtype of `home_goals` would otherwise depend on which rows the query
   selected, and code that worked on the full table would break on a subset.
   Re-asserting is also what makes a store read *equal* a `pd.read_parquet` of
   the same file, which is the property that makes the store an indirection
   rather than a behaviour change.
3. **Checks report; they do not gate the write.** The Parquet is written either
   way. A table you can open and inspect is more useful than one the pipeline
   refused to save, and severity — not the write path — decides what stops a
   caller.

### The leakage classification

Every canonical column now declares which side of kick-off it is knowable on,
and a check fails if any is unclassified. The reasoning matters more than the
lists: the defence has to be **temporal, not columnar**. `home_shots` is a
summary of the ninety minutes and using it for its own match is textbook
leakage — but a team's shots in its *earlier* matches are a legitimate feature,
so the columns are kept and the rule is about *when*, not *whether*. Odds are
genuinely pre-match and still excluded by default, for the separate reason that
a model trained on them copies the bookmaker.

Milestone 6 builds the temporal-integrity suite on top of this: a feature
declares which side it draws from, and CI fails any that can see its own match.

### What the gate found on day one

- **One row in 303,517 is filed under the wrong season.** An Argentinian match
  played 2015-01-29, labelled 2013-14 in the provider's own file, 213 days
  outside the widest defensible window. Reported as a warning and left in
  place: one misfiled row does not justify refusing the dataset, and silently
  carrying it into a season-level aggregate does not either.
- **The suite crashed on an empty table.** `min()` over an empty nullable
  column returns `pd.NA`, and `pd.NA < 0` raises rather than being falsy — so a
  validation report would have died at the moment it was most useful. Found by
  the empty-table test, not by review.
- **The feed capability flag is a ceiling, not a promise.** `ENG_5` sits in the
  primary feed, which declares `MATCH_STATS`, and supplies shot data for 2% of
  its recent matches. The aggregate check is set against the measured 93%, and
  the per-competition breakdown lives in the dataset card where it can be seen
  rather than averaged away.

### Changed from the approved scope, with reasons

The approved architecture is DuckDB + Parquet + PostgreSQL + MLflow + Docker
Compose. Two of those are **deferred, not dropped**, because building them now
would contradict this project's own standing rule against dead code:

- **PostgreSQL and `docker-compose.yml` → Milestone 11.** Postgres was scoped
  for "application data and prediction serving". At Milestone 3 there is no
  application data and no prediction. A Postgres store today is an unused
  second implementation of an interface with one caller, a schema with no rows,
  and a compose file nobody runs — three things a later milestone would have to
  rewrite anyway once it knew what a served prediction looks like. What makes
  the deferral safe is delivered: `MatchStore` is the seam, and CI enforces
  that nothing reads the table around it.
- **MLflow → Milestone 8.** It tracks runs. There are no runs.
- **pandera → not planned.** It was pencilled in for schema validation. Two
  thirds of these checks are semantic rather than structural — home advantage,
  a book's overround, a season's date window — and the structural third would
  need a pandera schema that is a second copy of `CANONICAL_SCHEMA` to keep in
  step. The one place a library would have earned its keep is the smallest
  part of the problem.

---

## Milestone 4 — Ratings ✅

**Delivered.** Two ratings, and the machinery that proves neither can see the
match it is rating.

| Module | Responsibility |
|---|---|
| `ratings/base.py` | The `RatingModel` protocol, the ratings schema, the chronological guard. |
| `ratings/elo.py` | One Elo pool per country, online, with home advantage, margin of victory, autocorrelation damping and season carry-over. |
| `ratings/dixon_coles.py` | Bivariate Poisson with the low-score correction and time decay, refitted per competition on a rolling window. |
| `validation/temporal.py` | Prefix invariance and outcome independence, generic over any derivation. |
| `validation/ratings.py` | The arithmetic and coverage checks on the output. |
| `pipelines/ratings.py` | Orchestration: build, probe, check, persist. |

**Verified:** 498 unit tests, 100% coverage of `src`, ruff/black/mypy clean,
plus 21 integration tests against the real table. The full build rates all
303,517 matches in about ten minutes — Dixon-Coles pricing 92.1% and Elo all of
them — with every causality probe holding and 9 of 9 checks passing, none
skipped. Both probes were checked against planted leaks before being trusted.

Full measurements are in `docs/RATINGS.md`. The rest of this section records
the decisions that changed.

### Causality is tested, not asserted

Ratings are the first quantity here with memory, and therefore the first place
a leak can hide. Reading the code for it does not scale — a rolling mean with
an off-by-one window, a normalisation over the whole table, a rating updated
before it is read: all look correct and all leak.

Two probes, and the pair is the point. A derivation that reads its own row is
*perfectly* prefix-invariant, because every value only ever depended on its own
row, so truncation changes nothing; only the scoreline rewrite finds it. Both
were verified against planted leaks of both kinds before being trusted.

They run on every ratings build against a sample competition, and the build
exits non-zero when one fails. The report says which competition — "verified"
and "verified on ENG_1" are different claims.

### Changed from the approved scope, with reasons

- **Per-competition Elo fitting: built, measured, removed.** The plan called
  for it. It made the ratings worse — 0.16300 against 0.16246 for fixed
  constants — because a calibration window is the coldest part of the history
  and three free parameters chase warm-up noise. A pooled fit was a wash. The
  constants come from one pooled search over pre-2005 matches, and
  `--fit-until` re-derives them when the data grows. Per-competition tuning may
  still pay against a *three-class* objective, which Elo cannot express; that
  belongs to Milestone 7, which owns the splits such a fit needs.
- **Ratings live in `data/features/`, not beside the canonical table.** The
  plan said "alongside". They are derived and rebuildable from `processed`,
  which is the definition `PathsConfig.features_dir` already carries; putting
  them in `processed` would have made the word meaningless.
- **Milestone 6's leakage suite starts here.** `src/validation/temporal.py`
  was scoped for Milestone 6. Writing it there would have meant Milestone 4
  shipping two rating models with their central property unverified for two
  milestones. It is generic, so Milestone 6 adds feature-specific checks rather
  than a second suite.

### One bug the tests found

`DuckDBStore.open_matches` attached the matches view, then failed on a missing
ratings file, and left a file-backed catalog holding half of what was asked for
— which the next open would have reported as success. Every source path is now
checked before the catalog is touched.

---

## Milestone 5 — Feature engineering ✅

**Delivered.** Twenty features, and three mechanisms that keep them causal
without relying on anyone reading the code correctly.

| Module | Responsibility |
|---|---|
| `feature_engineering/windows.py` | The causal primitive: every window ends at the last row *strictly earlier by date*. |
| `feature_engineering/registry.py` | What each feature reads, and therefore whether it can leak at all. |
| `feature_engineering/team_history.py` | Form, venue form, rest, congestion. |
| `feature_engineering/head_to_head.py` | Prior meetings, from the home side's end. |
| `validation/features.py` | 10 checks, including a table-level leakage gate. |
| `pipelines/derived.py` | The shape both derived-table pipelines share. |
| `pipelines/features.py` | Orchestration. |

**Verified:** 100% coverage of `src`, ruff/black/mypy clean, and 31 integration
tests against the real table. A full build is ten seconds; every probe holds.

Full measurements are in `docs/FEATURES.md`.

### The window cuts on the date, not the row

`shift(1).rolling(k)` is the obvious implementation and it is subtly wrong:
`shift` counts rows, so two matches on the same date let the earlier one —
earlier only by an arbitrary tiebreak in the sort — inform the later. Every
window here ends at the last row strictly earlier *by date*, which also makes
the builders survive the probes, since a truncation that removes a same-day
sibling then changes nothing.

Asking whether that mattered is what found the mislabelled-division bug: the
answer was 2,444 team-days, and all of them were the same fixture published
twice.

### Four defences, not one

1. The window cuts on date.
2. The registry derives `can_leak` from the declared `reads`, so there is no
   field to set wrongly. Seven of the twenty features read nothing but the
   fixture list and cannot leak whatever they do with it.
3. The probes recompute each builder over truncated and rewritten inputs, on
   every build, with a non-zero exit.
4. A table-level check: every window feature has a companion count, and a value
   where the count is zero came from somewhere it should not have. This one
   runs over every row written, not just the sampled competition the probes
   see.

### What the features are worth

Home win rate moves from 31.7% to 61.1% across the form-gap range, and the draw
rate peaks between evenly matched sides — which football says should happen and
the feature was not built to produce. Venue form is worth ±0.275 points per
game, symmetric to three decimal places between the two sides, which is the
check that the reshape is not confusing them.

**Rest days carry no marginal signal at all**: two points of spread and not even
monotone. Kept for the interaction rather than the marginal, and Milestone 8's
ablation is where it earns its place or is dropped.

### Changed from the approved scope

- **`src/pipelines/derived.py` was extracted.** The ratings and feature
  pipelines had the same shape — run producers, probe, check, persist, report —
  and written twice the two would drift, with the drifting half always being
  the probe nobody looked at again. The ratings pipeline was refactored onto it
  in the same change, with its tests unchanged.
- **Milestone 6 is now smaller.** Its temporal suite was written in Milestone 4
  and is exercised by every builder here, so what remains is extending it to
  whatever Milestone 7 derives, not building it.

---

## Milestone 6 — Leakage suite (next)

Scope, for approval:

- Extend `src/validation/temporal.py` to run over *every* registered producer
  as a CI step rather than only inside each pipeline, so a new builder cannot
  be added without being probed.
- A third probe for the split boundary: no training row may be dated after any
  evaluation row.
- A written audit of every column the model layer will see, tracing each back
  to the canonical columns it reads.
