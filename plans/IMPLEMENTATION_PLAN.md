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

## Milestone 6 — Leakage suite ✅

**Delivered.** Two more probes, and one change of principle: the suite stopped
taking a list of what to check.

| Module | Responsibility |
|---|---|
| `validation/leakage.py` | Finds every producer by walking the packages; runs both probes over each; traces every derived column to its inputs. |
| `validation/temporal.py` | Now four probes: prefix invariance, outcome independence, split boundary, observed reads. |
| `tests/unit/test_leakage_suite.py` | The suite, run on every test run and named as its own CI step. |
| `scripts/audit_columns.py` | `make audit` — the command behind `docs/LEAKAGE.md`. |

**Verified:** 100% coverage of `src`, ruff/black/mypy clean, 680 unit tests and
31 integration tests. Every probe holds over all four producers, on synthetic
data in the suite and on 2,400 real matches in the audit.

Full measurements are in `docs/LEAKAGE.md`.

### A list is a thing you can forget to add to

Milestones 4 and 5 each probe the producers on their own list. A builder wired
into a pipeline but omitted from its probe call would have shipped unverified,
and nothing in the output would have said so — the whole failure mode the
probes exist for, one level up.

So `producers()` does not take a list. It walks `src.ratings` and
`src.feature_engineering`, finds every class satisfying the contract, and
constructs each with the defaults the pipelines use — so what is probed is what
ships, not a test-tuned configuration that could hold while the shipped one
does not. `check_defaults_are_complete()` closes the other direction: a
producer nobody runs is not a leak, but it is a column the model layer expects
and will not get, which would otherwise surface milestones later as nulls.

### The third and fourth probes

**Split boundary** is the one the approved scope named. Ties fail, on the same
reasoning that makes every window cut strictly on the date: a full Saturday
programme is one round. Milestone 7 owns the splits, so the probe is tested and
waiting rather than wired.

**Observed reads** was not in the scope and is what made the audit worth
writing. Rewriting one input column and seeing which outputs move produces the
same table a hand-written audit would — except measured. It also gave the
feature registry the check it was missing: `reads` is hand-written, a typo in
it silently reclassifies a leaking feature as safe, and nothing until now
compared the declaration against behaviour. The assertion is coverage, not
equality — the measurement is a lower bound, and over-declaring is the safe
direction.

### What the audit found

Nothing wrong, and four things that reading the code does not tell you: Elo
reads `season` and never `date`; Dixon-Coles reads `competition_id` while Elo
reads no partition column at all, because its per-country pooling is already
inside the team id; `home_venue_points_5` does not read `away_team_id`, which
is the asymmetry a reshape is most likely to get wrong; and `elo_*_played`
reads no result.

It also found its own blind spot immediately. The first run, over ENG_1's
earliest 2,000 matches, reported that no feature reads shots — true of that
slice, because the provider carried no shot data before 2000/01, and a column
that is entirely null cannot be perturbed. The sample now takes the most recent
matches, and the limit is documented as a property of the sample rather than of
the code.

### Changed from the approved scope

- **"As a CI step" became a unit test that CI names as a step.** A shell step
  running its own probe script would have been a second way to run the same
  code, and the two would drift. The suite runs under `pytest`, so a producer
  added on a branch is probed by the command its author already runs locally,
  and the workflow calls it out separately so a failure appears in the job list
  rather than as one dot among five hundred.
- **A fourth probe, not a third.** The written audit the scope asked for would
  have been prose. Measuring it cost about twenty lines and turned the feature
  registry's declaration into a claim that fails when it is wrong.

---

## Milestone 7 — Splits and baselines ✅

**Delivered.** The first milestone that produces the number the whole project
is aimed at.

| Module | Responsibility |
|---|---|
| `models/splits.py` | Walk-forward folds, cut on the date, every one gated by the boundary probe. |
| `models/baselines.py` | Home-always, class prior, Dixon-Coles, the closing line. |
| `evaluation/metrics.py` | Log loss, RPS, accuracy — and what they refuse to score. |
| `pipelines/backtest.py` | Two subsets, per competition and pooled, written at the finest grain. |
| `scripts/backtest.py` | `make backtest`, five seconds. |

**Verified:** 100% coverage of `src`, ruff/black/mypy clean, 768 unit tests and
43 integration tests. Both new CI invariants hold.

Full measurements are in `docs/EVALUATION.md`.

### The headline

Five yearly folds, 62,036 evaluation matches, scored on the 59,001 every
forecaster could price:

| | log loss | RPS |
|---|---:|---:|
| Bookmaker closing odds | **0.9993** | **0.2031** |
| Dixon-Coles | 1.0277 | 0.2114 |
| Class prior | 1.0751 | 0.2284 |
| Home always | ∞ | 0.4316 |

**0.0284 of log loss** between the rating and the line, measured on identical
matches. The rating is 0.047 clear of the prior, so it is worth building; the
line is ahead everywhere, in all 39 competitions, which is the result a public-
data model should expect.

### Two findings the model milestone should start from

The rating's edge over the prior tracks the spread of team strength in a
competition at **r = 0.90** — a strength model has the most to say where
strengths differ most, which nobody told it to do. The gap to the closing line
tracks that same spread at **−0.14**, meaning what the bookmaker knows on top
of the rating is *not* strength, and is worth about the same amount everywhere.

That is the target stated as a number before a single feature is chosen: a
constant 0.028 that has nothing to do with how lopsided the league is.

Dixon-Coles also loses to the class prior in exactly one competition, the
Argentine cup. The obvious explanation — a cup mixes tiers — is contradicted by
the data: all 32 clubs also play in ARG_1, and the cup's Elo spread is the
seventh narrowest of 39. It is a narrow field fitted per competition on 610
matches while the same clubs' 2,211 league matches sit unused next door. Left
in; it argues for pooling a cup with its country's league, which is a change to
the rating and needs an ablation behind it.

### Changed from the approved scope

- **Metrics landed in `src/evaluation/`, which the plan gave to Milestone 10.**
  A baseline cannot be reported without them, and writing them in `src/models`
  to move them later is churn with a rename in it. Same precedent as Milestone
  4 writing the probes Milestone 6 owned.
- **Home-always is reported; the class prior is the floor.** The scope named
  home-always as a baseline to beat. As a probability forecast its log loss is
  infinite, so it is reported for what it does show — 43.7% accuracy, and what
  a proper scoring rule does to certainty — and the honest floor is the class
  prior counted per fold.
- **Two tables, not one.** The bookmaker prices 99.8% of these matches and
  Dixon-Coles 95.3%. Reporting a single table would have compared numbers
  computed over different sets of matches, which is what `docs/RATINGS.md`
  already refused to do once.

---

## Milestone 8 — Model zoo ✅

**Delivered.** Six families, and a result that is more interesting than the
ranking.

| Module | Responsibility |
|---|---|
| `models/dataset.py` | The thirty columns, derived from the two registries; the blocks the ablation withholds. |
| `models/zoo.py` | Six families, one wrapper, a fresh estimator per fold. |
| `models/tuning.py` | Optuna on matches strictly earlier than every reported fold. |
| `models/tracking.py` | MLflow to local SQLite, failure-tolerant. |
| `pipelines/train.py` | Training and ablation, both through the *unchanged* backtest. |
| `scripts/train.py` | `make train` (8 min), `make ablation` (5 min), `--tune`. |

**Verified:** 100% coverage of `src`, ruff/black/mypy clean, 880 unit tests and
61 integration tests. Every CI invariant holds.

Full measurements are in `docs/MODELS.md`.

### The headline

| 59,001 matches | log loss | RPS |
|---|---:|---:|
| Bookmaker closing odds | **0.9993** | **0.2031** |
| CatBoost | **1.0159** | 0.2083 |
| XGBoost / LightGBM / logistic regression | 1.0161–1.0162 | 0.2083 |
| Random forest | 1.0172 | 0.2087 |
| MLP | 1.0203 | 0.2091 |
| Dixon-Coles | 1.0277 | 0.2114 |

Every family beats the rating, in all 39 competitions. The best closes **0.0118
of the 0.0284** gap Milestone 7 measured, leaving 0.0166.

### Three findings for Milestone 9

**The top four are within 0.0003 of each other.** Logistic regression on thirty
columns is not distinguishable from three tuned gradient-boosting libraries.
Milestone 7 established that what the bookmaker knows on top of a strength
rating is not strength; this establishes that it is not a non-linear function
of these thirty columns either. What is left is missing information, not
missing capacity — which is a statement about where *not* to look next.

**Form is worth more than either rating.** Withholding the fourteen form
columns costs 0.0033; Elo costs 0.0021 and Dixon-Coles 0.0016, each alone,
because the two ratings are substitutes. Rest days and congestion are worth
0.0003 — the question Milestone 5 left open, now settled: they survive and are
the first thing to trim. Head-to-head is worth 0.0001.

**The zoo fixed the competition the rating could not price.** Dixon-Coles lost
to counting base rates on the Argentine cup; LightGBM gets 1.0806 there against
the prior's 1.0902, with no special case. Milestone 7 asked whether the cup
needed a pooled Dixon-Coles fit. It needed a model around the rating instead,
and that is one fewer change to a component two milestones downstream now
depend on.

### Changed from the approved scope

- **Search budgets differ per family, and are stated rather than implied.**
  Twenty trials for the two cheapest, three for the most expensive; a trial is
  three fits over most of the history. Uniform budgets would have meant either
  three trials each or several hours, and the evidence — four of six searches
  moving the fourth decimal — did not justify the hours.
- **The tree search spaces were narrowed once, after measurement.** LightGBM's
  first version reached 255 leaves and 800 estimators, and one trial there took
  longer than XGBoost's entire search, for a model nobody would ship on thirty
  tabular columns. The module already claimed its spaces were "deliberately
  narrow"; this made that true. Every reported search used the narrowed space.
- **No ensemble was built, deliberately.** Averaging the top four is the
  obvious next move and belongs in Milestone 9 with the calibration. Built
  here, on four models within 0.0003 of each other and heavily correlated, it
  would report a number that says more about the averaging than the models.

---

## Milestone 9 — Ensembling and calibration ✅

**Delivered.** Both layers built as scoped, and the reportable answer the scope
allowed for is the one that came back.

| Module | Responsibility |
|---|---|
| `models/ensemble.py` | The blend, its members chosen on error correlation, and the diagnostic pass that gets the matches back. |
| `models/calibration.py` | One temperature, fitted on a holdout inside the training half. |
| `evaluation/reliability.py` | Whether a stated probability happens at the rate it states. |
| `pipelines/train.py` | `run_ensemble` and `reliability_tables`, both through the *unchanged* backtest. |
| `scripts/train.py` | `make ensemble` (~20 min), `make correlations` (~3 min). |

**Verified:** 100% coverage of `src`, ruff/black/mypy clean, 930 unit tests and
68 integration tests. Every CI invariant holds.

Full measurements are in `docs/MODELS.md`.

### The headline

| 59,001 matches | log loss | calibration error |
|---|---:|---:|
| Bookmaker closing odds | **0.9993** | — |
| Blend of three, calibrated | **1.01560** | **0.0015** |
| Blend of three | 1.01565 | 0.0045 |
| CatBoost, Milestone 8's best | 1.01589 | — |
| XGBoost, calibrated | 1.01622 | 0.0020 |
| XGBoost | 1.01614 | 0.0037 |

Both layers together close **0.0003** of the 0.0165 Milestone 8 left. That is
the milestone's result, and it is worth stating as a result rather than
apologising for: two layers that a great many projects report as a headline
improvement are, on this problem and against a properly measured baseline,
worth about a fortieth of what the model layer itself was worth.

### Three findings

**Choosing members on error correlation picked a blend nobody would have
picked.** The four tree-based families correlate between 0.9934 and 0.9960 —
substitutes, which is the quantitative form of Milestone 8's "the top four are
within 0.0003". The rule admitted XGBoost, logistic regression and the **MLP**,
the worst model in the zoo, because it is the only one wrong about different
matches at 0.92. That blend beats every family that went into it *and*
CatBoost, which did not. Averaging the top three would have averaged three
near-substitutes and reported the averaging.

**Calibration buys reliability and not loss.** The scalar halves the gap
between what the model states and what happens — 0.0037 to 0.0020 — and moves
log loss by 0.00008, in the wrong direction. Milestone 8's models were fitted
on a proper scoring rule, so they were already close to proper; what was left
was a systematic overconfidence in the 0.4–0.5 band that log loss barely
charges for and a reliability table shows at a glance.

**The remaining gap is looking less like a modelling problem every
milestone.** Milestone 7 measured 0.0284 and said what the bookmaker knows on
top of a strength rating is not strength. Milestone 8 closed 0.0118 and said it
is not a non-linear function of these thirty columns. This one closes 0.0003
with the two standard moves that are left, which is the strongest evidence yet
that the last 0.0163 is information this project does not have.

### Changed from the approved scope

- **The calibration layer is measured over XGBoost, not CatBoost.** The scope
  said "the best single model". CatBoost is best on the reported folds by
  0.0002; XGBoost is best on the tuning slice. Choosing the subject of a
  measurement by looking at the folds it is then measured on is the mistake the
  member selection is arranged to avoid, so the constant follows the slice —
  and the difference is smaller than the one Milestone 8 called
  indistinguishable.
- **Reliability costs a second pass over the folds.** The backtest persists
  means, by design, and reliability needs the matches back. The alternative was
  a side channel out of the scoring pipeline; the pipeline's ignorance of what
  it is scoring is the reason a model and a baseline can share a table, and it
  was worth more than the minutes.
- **The blend is scored beside the same four baselines rather than beside the
  whole zoo.** The two runs share a common subset — both include the two
  baselines whose coverage defines it — so the tables are comparable across
  runs and re-fitting five families to reprint Milestone 8's numbers would have
  bought nothing. There is an integration test asserting the two subsets match.

---

## Milestone 10 — Evaluation and explainability ✅

**Delivered.** All three items, and the two methods disagreed — which the scope
said would be the finding, and it was.

| Module | Responsibility |
|---|---|
| `explainability/permutation.py` | Break a block at prediction time; covers every family. |
| `explainability/shapley.py` | `TreeExplainer` over the boosted three, aggregated to blocks. |
| `evaluation/model_card.py` | The card, rendered from data it is handed. |
| `pipelines/report.py` | Reliability per class and per competition; the card assembled. |
| `pipelines/tables.py` | The three tables, joined once for the three commands that need them. |
| `scripts/explain.py`, `scripts/model_card.py` | `make explain` (~2 min), `make card` (~5 min). |

**Verified:** 100% coverage of `src`, ruff/black/mypy clean, 1,021 unit tests
and 83 integration tests. Every CI invariant holds, plus a new one keeping
`src/explainability` above the storage and pipeline layers.

Full measurements are in `docs/EXPLAINABILITY.md`; the card is
`docs/MODEL_CARD.md`.

### The headline

LightGBM on the most recent fold — 291,715 training matches, 11,802 scored:

| Block | breaking it | share of the arithmetic | never having had it |
|---|---:|---:|---:|
| `elo` | **+0.0395** | **38.0%** | +0.0021 |
| `form` | +0.0087 | 37.8% | **+0.0033** |
| `dixon_coles` | +0.0073 | 19.9% | +0.0016 |
| `schedule` | +0.0001 | 2.4% | +0.0003 |
| `head_to_head` | +0.0001 | 2.0% | +0.0001 |

### Three findings

**The methods disagree about the top two, and that is the result.** Permutation
and SHAP rank elo first; the ablation ranks form first. The model reaches for
elo hardest and can most easily do without it, because Dixon-Coles is a
substitute — Milestone 8's "the two ratings are substitutes", now a 19× gap
between what a block is *used for* and what it is *worth*. Form's two numbers
agree because nothing substitutes for it. The practical reading: a block whose
numbers diverge has a backup, and a block whose numbers agree is load-bearing.

**Where all three agree, they agree completely.** `schedule` and
`head_to_head` are at 0.0001–0.0003 by every method on every family. Three
independent measurements at the noise floor is the strongest statement this
project has made about a feature block, and it closes what Milestone 5 left
open.

**The shipped model is most honest about draws and least useful there.** Per
class, the calibrated blend scores 0.0012 on draws against 0.0035 on home wins
— because it states about a quarter every time, and about a quarter of matches
are drawn. Reliable is not the same as useful; the class prior is perfectly
reliable and worthless, which is why the card carries the score table beside
the reliability one. Per competition, the Argentine cup is the least reliable
of the 39 at 0.0270 against a pooled 0.0015 — the same competition Dixon-Coles
could not price in Milestone 7.

### Changed from the approved scope

- **SHAP covers three families of six, not the blend's members.** The scope
  asked for SHAP over the blend's members; two of those three are scikit-learn
  pipelines that `TreeExplainer` refuses, and the general-purpose explainer
  costs hours per family for a number permutation importance produces exactly
  in seconds. So the pair is SHAP where it is exact and permutation everywhere,
  and the comparison against the ablation — which was the point — is made on
  the family the ablation was run on.
- **`matplotlib` was dropped.** It was listed for this milestone in
  `requirements.txt` and is not installed: every figure this work would have
  drawn is a five-row table, and a PNG in a repository is a number that goes
  stale without a diff to show for it.
- **`TrainedForecaster` gained `fit()` and `predict()`.** Behaviour-preserving,
  and required: SHAP needs the fitted estimator and permutation predicts
  nineteen times off one fit. Both would otherwise have reimplemented the
  class-scattering that keeps a forecast in H/D/A order — the bug Milestone 8's
  test suite calls the most likely silent one in the project.
- **The attribution runs on one fold, not five.** A ranking whose gaps are an
  order of magnitude apart does not move on a second fold, and the ablation it
  is compared against is pooled over five — stated here rather than implied.

---

## Milestone 11 — API ✅

**Delivered.** The shipped blend, fitted once, behind HTTP, in a container.

| Module | Responsibility |
|---|---|
| `models/artifact.py` | The blend as a *fitted* object: members, the mean, one temperature. Frames in, arrays out. |
| `pipelines/serving.py` | Persisting it with a manifest, reading it back with the checksum checked first, and the fixture index. |
| `storage/predictions.py` | The served-prediction log: PostgreSQL, and the no-op that stands in when none is configured. |
| `api/service.py` | What answers a request: one model, one index, one log. |
| `api/routes.py` | Six endpoints, each a call and a return. |
| `api/main.py` | Lifespan, access logging, and the four exceptions mapped to status codes in one place. |
| `api/schemas.py` | The request and response models, which are also the OpenAPI document. |
| `scripts/build_model.py` | `make model`. |

**Verified:** 1,131 unit tests, 100% coverage of `src` **and** `api`,
ruff/black/mypy clean, plus integration suites against the real tables and
against a real PostgreSQL. CI gained two invariants and a job that builds the
image and curls `/health` against it.

Full API reference in `docs/API.md`. The rest of this section records the
decisions that changed.

### The one design decision the plan did not anticipate

The scope said "one fixture in". It did not say where the fixture's thirty
columns come from, and there are only two answers: look them up, or recompute
them.

**Recomputing was rejected, and the data settles it.** The provider publishes
*results*, not a fixture list — there is no feed of next Saturday's matches
anywhere in this project — so a "price this upcoming match" endpoint has
nothing to be handed. And even with such a feed, a design row built inside a
request handler would be a second implementation of the feature layer, living
outside every probe `src/validation/leakage.py` runs on the first. Milestone 6's
whole argument is that a leak has no symptom: it simply makes the model look
better. So the service reads the row the audited pipeline wrote, and says so.

What that costs is on every response rather than in a footnote. `make model`
fits on the whole history, so a fixture in the table is a fixture the artefact
trained on; `in_sample` is what keeps that probability from being quoted as the
walk-forward one.

### The artefact is a new object, not a serialised old one

`TrainedForecaster` holds its `build` as a lambda — the right shape for a zoo
that wants a fresh estimator per fold, and unpicklable. `FittedMember` stores
the family *name* and its columns instead and rebuilds the wrapper from the zoo
on demand, so the scatter that puts `predict_proba` back into `CLASSES` order
stays one implementation. An artefact naming a family the zoo no longer has
fails on load, naming it.

### Changed from the approved scope, with reasons

- **SQLAlchemy: not installed.** `requirements.txt` pencilled it in beside
  psycopg. There is one table, created if absent, with no column ever dropped
  or renamed; an ORM plus a directory of versioned migrations would be a second
  description of a table that fits on a screen. psycopg alone, every value
  bound, and the connection injected so the unit suite drives every branch
  offline — the same arrangement `src/utils/http.py` already uses for a stubbed
  session.
- **The prediction log is optional.** The plan implies it is always there. A
  clean checkout has no PostgreSQL and neither does CI, and an API that would
  not start without one is an API nobody can try. `PREDICTION_LOG_DSN` unset
  means a no-op log, and `/health` says so rather than leaving it to be
  inferred from silence.
- **A `/fixtures` endpoint was added.** Not in the scope, and without it
  nothing else in the service is usable: the fixtures that exist are the ones
  the batch build wrote, and a caller has no other way to learn which those
  are.
- **`docker-compose.yml` arrived here rather than in Milestone 13.** Milestone
  3 deferred it on the grounds that there was no application state and nothing
  to serve. Both now exist, and a Postgres service with no compose file is a
  dependency nobody can run.

### Two bugs the image found that review did not

Both were `ModuleNotFoundError` at container start, and neither is visible in a
diff.

`src/pipelines/tables.py` imported `MATCHES_FILENAME` from
`src/pipelines/ingest.py`, which pulls the provider adapter, the registry, the
cache and `requests` — so the serving process loaded the whole ingestion stack
to read one string. The constant now lives in `src/ingestion/base.py`, with the
canonical schema it names, and `ingest` re-exports it.

`src/models/zoo.py` imported all six families at module scope. The served blend
contains one boosted library; at module scope the image had to carry all three
to satisfy an `import` no request reaches. Each is now imported when its family
is built.

---

## Found by audit, awaiting a milestone — fourteen candidate features

Not scheduled, and not lost. A full audit measured fourteen columns built from
canonical fields the design matrix does not read — half-time scores, shots on
target, a 20-match form window, `tier`, and days into the season — against the
shipped thirty on the same folds with the same tuned XGBoost:

| | log loss | |
|---|---:|---|
| base 30 columns | 1.01690 | |
| plus 14 candidates | **1.01651** | +0.000394, p = 0.0021, n = 62,036 |

Reproduced across three seeds at five to eight times the seed-noise floor
(0.00005), so it is an effect rather than a refit. `tier` and `season_days`
carry 0.000219 of it between them at p = 0.018 — the model currently has **no
competition-level context at all**. Both leakage probes pass.

It is not in this plan's numbered sequence because adopting it regenerates
every reported number in the repository, which is a milestone with its own
ablation rather than a patch. The working, the harness and the honest caveats
are on the **`experiment/candidate-features`** branch:

```bash
git checkout experiment/candidate-features
python -m experiments.run_experiment    # ~8 min, reproduces every number above
```

That branch also records the one candidate that leaked and was caught by
`prefix_invariance` on the first run, which is the best argument this
repository has for probing a derivation rather than reading it.

---

## Milestone 12 — Dashboard ✅

Delivered, in two passes. The first was four tabs over the reporting layer. The
second turned it into the presentation layer of a football application: six
pages over four layers, with the seams the next eight milestones plug into.

### The architecture, and what each future milestone costs because of it

```
views  ──▶  services  ──▶  providers  ──▶  domain
```

- **`domain/`** — `Fixture`, `Prediction`, `MatchStatus`, the competition
  catalogue, favourites. Value types with no I/O.
- **`providers/`** — one protocol per kind of source, one implementation per
  source. `HistoricalResults` (the canonical table), `ApiPredictions` (the
  service, over HTTP), `NullFixtures` (no feed yet, and the reason).
- **`services/`** — orchestration and caching. `matchday.home_page` assembles
  four sections from three providers and a favourites list; `history` holds the
  Streamlit caches and the three small tables a fixture page shows.
- **`views/`** — Streamlit. Thin, and the only layer a different front end
  would replace.

| Milestone | What it costs, given this shape |
|---|---|
| 13 — live fixtures | One class satisfying `FixtureProvider`, one registry entry, one environment variable. No view moves. |
| 14 — accounts | Four accessors in `domain/favourites.py`. No view touches `st.session_state`. |
| 15 — real-time | A second method on the fixture feed; the card already carries a minute and a score. |
| 17 — odds, xG | A fourth protocol beside the three, and a section on the match page that already has its placeholder. |
| React client | `views/` is replaced; `domain`, `providers` and `services` are reused. This is why the caching is in the services and the providers hold no state. |

CI asserts the arrows, including the one that matters: **no view names a
provider.**

### The open question in the original scope, answered

- **"A client of `api/` *or* of `src/pipelines`"** — the answer turned out to be
  *three* paths, because each answers a different question. Results come from
  the match table (the API deliberately serves no scoreline). Measurements of
  the model come from the report tables (they already exist; recomputing them
  would be a second number to reconcile with the card). Forecasts come over
  HTTP (two processes that unpickle the artefact are two implementations of
  "what does the model say").
- **The page works with the service down, and with no data at all.** That falls
  out of the split rather than being designed for. Each section names which of
  the three is missing and the command that produces it.

### The honest gap

Today's matches, upcoming fixtures and live scores are **not** on this page,
and the reason is data rather than design: this project ingests results, so a
match that has not been played is in no table here. Those sections render from
a real implementation of the interface that returns nothing and states why.

Nothing invents a fixture. A plausible generated match is a game on the screen
that is not being played, and a reader who catches that once stops believing
the real rows too.

### The enabling changes

- `make card` computed 62,036 per-match forecasts, rendered the card from them
  and threw them away. It now writes
  `data/reports/ensemble/forecasts.parquet` with a manifest beside it, so the
  dashboard reads them instead of spending five minutes on every page load.
- `src/pipelines/tables.py::read_matches` — finished matches, projected to the
  twelve columns a reader is shown. The only other change to `src` this
  milestone made, and it reads: no model, feature, split, metric or reported
  number changed.

### One figure, and the Milestone 10 note it reverses

`requirements.txt` recorded at Milestone 10 that matplotlib was left out
because every figure that milestone would have drawn was a five-row table. The
reliability diagram is the exception and the reason is narrow: its claim is a
diagonal, `y = x` *is* the hypothesis being tested, and a reader checks a
forecast against it by eye in a way a column of signed gaps does not support.
Plotly rather than matplotlib because it is drawn in the browser, so nothing is
committed as a PNG that goes stale without a diff.

Everything else on the page is CSS: probability bars, crests, form strings and
cards are markup, not figures, because forty Plotly figures behind one scroll
is forty renders of a charting library.

---

## Milestone 13 — Live fixtures ✅

The milestone Milestone 12's shape was a bet about, and the bet paid: **one
class, one registry entry, one environment variable.**

```python
# dashboard/providers/__init__.py
FIXTURE_PROVIDERS = {"none": NullFixtures, "football-data.org": FootballDataOrgFixtures}
```
```bash
export FOOTBALL_DATA_API_KEY=...            # free: football-data.org/client/register
export DASHBOARD_FIXTURE_PROVIDER=football-data.org
```

`dashboard/providers/football_data_org.py` reads `GET /v4/matches` over a date
window and turns the rows into the `Fixture` the cards have rendered since
Milestone 12. Today's matches, the coming week and live scores — the three
things the plan recorded as absent for a data reason rather than a design one —
now have a source. Crests fill the two fields `ui.crest_html` has been drawing
initials for.

**No view moved**, which was the claim. One caption did: the sidebar read
`✕ No fixture feed — Milestone 13`, and a status bar citing an unshipped
milestone after it ships is a small lie a reader stops checking the rest of the
page against. It now names the connected feed, or the provider's own reason for
having nothing.

### Where the honesty had to be spent

- **It is not a second ingestion source.** Nothing this feed returns is written
  to a table, joined to one, or read by a model. No model, feature, split,
  metric or reported number changed this milestone, and no row of
  `matches.parquet` came from anywhere new.
- **Its ids do not pretend to be canonical.** `fdorg-497821`, not
  `make_match_id(...)`. A match id in this project hashes the natural key
  *including the team names as its source spells them*, and this feed says
  "Manchester United FC" where the ingested table says "Man United". An id that
  looked canonical and joined to nothing would be worse than one that names
  where it came from. `src/ingestion/teams.py::ALIASES` is therefore still
  empty: an alias table with no reader would be a guess, and the milestone that
  earns it is one that *ingests* a second source.
- **The visible consequence, stated rather than hidden.** A card from this feed
  opens a match page with no table row and no forecast, and that page already
  said so before this milestone existed. The shipped model is fitted on
  finished matches; a fixture that has not been played has no history to price.
- **Nine competitions of thirty-nine**, which is the free tier. A followed
  competition outside them is dropped from the filter, because there is no code
  here to send for it, and a reader who follows *only* such competitions is
  told that in a sentence.
- **A rejected key does not read as "no matches".** `available` makes the
  same window request `live()` makes, so a refused key renders as a key problem
  — in the feed's own words, *"Your API token is invalid."* — rather than as a
  claim that no football is scheduled this week. The memo means the two are one
  call.

### What the live API said that review did not

The provider was written against the published v4 contract and reviewed
against it. A smoke test with a real key found three things a reading could
not, which is the entire argument for spending the ten minutes:

- **Two clocks.** The feed indexes by UTC; `matchday` asks about
  `date.today()`, which is the host's. This machine runs at UTC+05:30, where
  those are *different days* for five and a half hours out of every
  twenty-four — so `live()` asked the feed for tomorrow and would have gone
  blank through Saturday evening in Europe, and tonight's late kick-offs were
  outside the "today and next seven days" window. Fixtures now carry host-local
  date and kick-off, the live window is anchored to UTC and filtered by
  *status* rather than by date, and a requested window is widened a day at each
  end and narrowed back in the answer. The regression test pins `TZ` to
  `Asia/Kolkata`, because in UTC — which is what CI runs in — the bug does not
  reproduce.
- **A ten-day cap.** `dateFrom`/`dateTo` more than ten days apart is answered
  `400 Specified period must not exceed 10 days`. A fourteen-day ask therefore
  returned nothing. Longer windows are now split into consecutive requests
  rather than truncated: a page missing fixtures with nothing on it saying so
  is the failure this provider layer is arranged against.
- **`dateTo` is an instant, not a day.** The window is
  `[dateFrom T00:00Z, dateTo T00:00Z]` and includes both ends, so `09-05..09-05`
  answers *nothing at all* and `09-05..09-06` answers the whole of the 5th plus
  anything kicking off at exactly midnight on the 6th. The live window was
  built as "yesterday to today", which under that rule excludes today
  entirely — `live()` returned nothing while two matches were in play, and it
  would have done so permanently. Every date in the module is now an inclusive
  day with one conversion at the wire.
- **Consecutive chunks share an instant**, which the first fix turned into a
  duplicate: a match at exactly midnight UTC — every Brazilian evening
  kick-off — is returned by both the chunk that ends there and the one that
  begins there. Chunks are merged by `match_id` rather than concatenated.
- **The 400s explain themselves in the body, not the status line.** This feed
  sends an empty HTTP reason phrase, so an invalid token and an over-wide
  window both read `answered 400: ` — the whole useful half is the JSON
  `message`. It is now what a reader is shown: *"Your API token is invalid."*

One documented claim was also wrong and is corrected: a paid competition left
in the filter is *ignored* (`competitions=PL,BL2` answers 200 with the PL rows),
not refused with a 403. Unsupported ids are still dropped, for the simpler
reason that this project's ids are its own and there is no code to send.

One more thing the feed does that this provider deliberately does not correct:
its status can lag. A 16:45 kick-off was still `IN_PLAY` at 22:40 with a
current `lastUpdated`. Inferring "that must have finished" from the clock would
be the dashboard inventing a result, so the feed's status is rendered as given.

Measured on the live feed with every fix in: **126 fixtures** over the coming
week across all nine competitions, **2 matches in play** with their scores,
**22** Premier League fixtures over fourteen days (two chunks, no seam
duplicate), **113 of 113** finished rows in a past window parsed with scores,
crests on every row, an empty far-future window answered as empty with no
error, a repeat call served from the memo in 0.2 ms, and an invalid key
reported as *"Your API token is invalid."* The home page renders **125 cards**
with no exception and the shell reads `✓ Fixture feed · football-data.org`.

### The one thing that is not in the class

Ten requests a minute, against a Streamlit script that reruns on every click.
Answers are memoised for sixty seconds by an `lru_cache` keyed on a time
bucket — module level, because the context builds a fresh provider on every
rerun and an instance cache would be empty every time it was read. That is the
whole of the caching: the stdlib already holds, bounds and evicts the entries.

---

## Milestone 14 — Accounts and saved favourites ✅

Milestone 12 wrote down what this would cost: *"four accessors in
`domain/favourites.py`. No view touches `st.session_state`."* It cost exactly
that. Every signature in `favourites.py` is the one it shipped, six views are
untouched, and what changed underneath is where the answer comes from.

Favourites used to live in `st.session_state` — per browser tab, gone when it
closed. They now live in a JSON file keyed by whoever the reader is.

### Two ways to be someone, and the second is optional

`dashboard/domain/identity.py` is the only module in the application that knows
how a reader is named:

| | Key | Needs |
|---|---|---|
| A profile | `profile:<name>` | Nothing. Picked from the sidebar, `Guest` by default |
| An account | `account:<verified email>` | An `[auth]` section in `secrets.toml` **and** `streamlit[auth]` |

`Guest` is a real profile with a real row rather than a null case, which is the
user-visible point: favourites persist for a reader who never opens the picker.

The keys are namespaced because without the prefixes, a profile named after a
colleague's email address would be handed that colleague's favourites. Small
here; the same shape is serious in an application that stored anything worth
taking.

### What the Streamlit auth surface actually does

Checked rather than assumed, which is the habit Milestone 13 paid for:

- **`st.user.is_logged_in` raises `AttributeError`** when no provider is
  configured — Streamlit adds the key only when `secrets.toml` has an `[auth]`
  section, and with none the object has no attributes at all. Everything here
  reads it through `.get`, and the key's *presence* is how the sidebar knows
  whether a provider exists.
- **`st.secrets` raises** outright when there is no secrets file, so "is auth
  configured" is never answered by reading it.
- **`st.login()` raises `StreamlitMissingAuthlibError`** without
  `streamlit[auth]`, which `requirements-dashboard.txt` deliberately does not
  install. The button appears only when both halves are present; a button that
  always errors is worse than no button.

### A file, with the ceiling written next to it

`data/dashboard/profiles.json`, or `$DASHBOARD_PROFILE_STORE`. Not the
PostgreSQL that is already in the compose file: that one is the *service's*,
holding served predictions so calibration can be measured against what
happened, and reaching it from here would mean `psycopg` in an image whose
requirements file documents its absence, a connection pool nobody tracks across
Streamlit reruns, and a schema migration for a preference.

One JSON file, rewritten whole, last writer wins — two people editing different
profiles in the same second lose one edit. The upgrade is that database and the
seam for it is `store.py`, which is the same trade Milestone 12 made and the
reason this one was cheap.

Writes go through a temporary file in the target's own directory and an atomic
rename, because a partial JSON document is unreadable and the failures that
produce one — a process killed mid-write, a disk filling — are exactly when a
reader can least afford to lose the file. A write that cannot land at all is a
sentence in the sidebar rather than an exception: `data/` is mounted read-only
in the compose file, so it is a state a real deployment reaches.

### The invariant that made this cheap, now enforced

`session_state` is used in exactly one module, asserted by CI and confirmed to
bite by planting a violation in a view. Milestone 12 claimed the property;
until now nothing checked it, and it is the reason accounts were three new
files instead of a search through six pages.

One test isolation bug is worth recording because it was found the honest way —
by it happening. The first suite run wrote real profiles into the developer's
`data/` directory, and every later test inherited whichever clubs an earlier
one had followed. `tests/conftest.py` now points the store at a temporary file
for every test in the suite, autouse: the failure is silent in both directions,
and a store that a test can reach is a store a test will write to.

---

## Milestone 15 — Real-time tracking and notifications ✅

Milestone 12 estimated this as *"a second method on the fixture feed; the card
already carries a minute and a score."* The card estimate held — no card
changed. The method did not: the feed already had `live()`, and what was
missing was **memory**. A page that can say what is in play cannot say what
*changed* without remembering what it said last time.

### Three parts

- **`dashboard/services/watch.py`** — one snapshot per browser tab, and the
  diff between two looks. `st.session_state`, because "since *I* last looked"
  is a per-tab question and a snapshot in the profile store would mean the
  first tab to refresh silently consumed the second one's news.
- **`dashboard/providers/webhook.py`** behind a new `Notifier` protocol, with
  `NullNotifier` as the default — the same registry-plus-environment-variable
  shape as the fixture feed, because "which transport is configured" and
  "which feed is configured" are the same question asked of different things.
- **`@st.fragment(run_every=60)`** on the live strip. A fragment rather than a
  whole-page rerun: everything else on that page is a file read or an HTTP
  call, and repainting all of it every minute to move one score would be the
  most expensive way to show the cheapest change. The interval is *matched* to
  the feed's memo rather than chosen, so a refresh that finds nothing new costs
  no request at all.

### The two rules that are wrong in the obvious implementation

- **The first look announces nothing.** With no previous snapshot there is no
  "since", and a toast reading "kick-off" for a match already an hour old is a
  page telling a reader something untrue.
- **An empty answer is not full time.** The feed returns nothing both when
  nothing is in play and when it could not be reached, so absence read as "the
  match ended" would announce eight final whistles because of one rate limit.
  The feed is asked whether it thinks it answered, and a failed look produces
  no events at all.

A changing *minute* is deliberately not an event — a notification per minute of
a match is a notification a person turns off — which is also why the free
tier's missing `minute` field costs this milestone nothing.

Full time turned out to be a match that has **gone** from the answer rather
than one whose status changed, because `live()` returns what is in play. That
is why a snapshot holds fixtures rather than a status-and-score string: by the
time a match is over, the snapshot is the only record of the score it finished
on.

### What was cut, and where the line is

**Alerts exist while something is watching.** The page has to be open. A
process that polls with every browser closed is a different thing with its own
lifecycle — a second consumer of a ten-request budget, reading favourites from
outside Streamlit — and it was left out rather than half-built. Everything it
would need is here: the diff is a pure function over two lists, and the
transport is a class with one method.

### Verified against real matches

Two matches genuinely in play on the live feed (Alavés v Osasuna, Vitória SC v
Casa Pia). The first look reported both and announced nothing. Rewinding the
snapshot by one goal — the feed will not score on demand — produced exactly one
`goal` event, delivered to a local webhook receiver; removing the match from the
answer produced one `full-time` carrying the score it finished on. The body a
receiver gets carries `text` for Slack, `content` for Discord and the event's
own fields for anything programmatic, so there is no "which flavour of webhook"
setting to answer.

Two dead branches were deleted rather than tested: a "was scheduled, is now
live" case that a snapshot of `live()` answers can never contain, and — from
the earlier draft — a full-time-by-status-change that the same fact rules out.
A branch no test can honestly reach is a branch that should not exist.

---

## Milestones 16–20 — the platform

The dashboard is the first milestone whose *shape* is a commitment about the
ones after it. These are the ones it was shaped for. Milestones 13, 14 and 15
above are spent, and the shape held each time — a provider class, four
accessors, and a service plus a transport.

| | | Where it plugs in |
|---|---|---|
| 13 ✅ | Live fixture ingestion | Done — `dashboard/providers/football_data_org.py` behind `FixtureProvider` |
| 14 ✅ | Accounts and saved favourites | Done — `dashboard/domain/{identity,store}.py` behind the same four accessors |
| 15 ✅ | Real-time tracking and notifications | Done — `services/watch.py` diffs the feed; `providers/webhook.py` behind a `Notifier` |
| 16 | Cloud deployment, monitoring, caching | The image and compose file that already exist |
| 17 | Bookmaker odds, expected goals, value detection | A fourth provider protocol; the match page has the placeholder |
| 18 | Player availability, injuries, transfers | A fifth; likewise |
| 19 | Historical prediction archive | `src/storage/predictions.py` already logs every served forecast |
| 20 | The platform | — |

Research models — TabNet, FT-Transformer, AutoML, benchmarked against the best
GBDT with a written verdict — is unscheduled rather than dropped. It changes
every reported number in this repository and belongs to a milestone with its
own ablation, not to a platform release.
