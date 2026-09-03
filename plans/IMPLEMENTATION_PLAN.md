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

## Milestone 8 — Model zoo (next)

Scope, for approval:

- Logistic regression, random forest, XGBoost, LightGBM, CatBoost and an MLP
  over the twenty features plus the ratings, scored on the folds this milestone
  built and against the baselines it measured.
- An ablation per feature group, which is where rest days earn their place or
  are dropped, and where a cup-shaped fixture list earns Dixon-Coles a pooled
  fit or a note saying not to use it there.
- Optuna for tuning and MLflow for the runs, now that there are runs.
