# Implementation plan

The decisions that shape every milestone, recorded once so they are not
re-litigated later. Milestone sections are filled in as each is delivered.

---

## Approved decisions

| # | Decision | Chosen | Rationale |
|---|---|---|---|
| 1 | Data provider | football-data.co.uk only, behind an adapter interface | Free, keyless, static CSV, no rate limit. Measured after a full ingest: 39 competitions, 305,499 matches, 1993-2026. Verified live before committing to it. Other providers plug in later without downstream change. |
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
one canonical 35-column schema. A full ingest produced 305,499 matches and
1,363 teams spanning 1993-2026, with zero unreadable files. No downstream module can tell which feed a
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

## Milestone 3 — Storage and validation (next)

Scope, for approval:

- `src/storage/base.py` — the storage protocol; DuckDB and Postgres behind it.
- `src/storage/duckdb_store.py` — the analytical store; Parquet feature store.
- `src/storage/postgres_store.py` — application data and prediction serving.
- `src/validation/` — schema checks, referential integrity, distribution
  sanity. The integration assertions written in Milestone 2 move here and
  become a first-class, reportable gate rather than a test file.
- A dataset card generated from the real ingest, with per-competition coverage.
- `docker-compose.yml` for Postgres.

**No feature engineering yet.** Milestone 3 makes the canonical table
queryable, validated and reproducible.
