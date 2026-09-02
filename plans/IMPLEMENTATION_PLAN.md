# Implementation plan

The decisions that shape every milestone, recorded once so they are not
re-litigated later. Milestone sections are filled in as each is delivered.

---

## Approved decisions

| # | Decision | Chosen | Rationale |
|---|---|---|---|
| 1 | Data provider | football-data.co.uk only, behind an adapter interface | Free, keyless, static CSV, no rate limit, ~38 competitions, ~250k matches. Verified live before committing to it. Other providers plug in later without downstream change. |
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
  available at match level across 38 competitions from any free source. Feature
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

## Milestone 2 — Ingestion (next)

Scope, for approval:

- `src/ingestion/base.py` — the `MatchProvider` protocol and the canonical
  match schema every adapter must produce.
- `src/ingestion/football_data.py` — both schemas (main and extra) behind one
  adapter, with per-competition capability flags.
- `configs/leagues.yaml` — the competition registry. Adding a league is a
  config entry, not code.
- Caching and freshness: a completed season never re-downloads; the current
  season refreshes on an age check. Checksum manifest for reproducibility.
- Team-name normalisation across seasons and schemas — the first real data
  problem, and the one most likely to corrupt joins silently.
- `data/sample/` — a committed slice, so the suite stays offline.
- `scripts/fetch_data.py` — the CLI entry point.

**No feature engineering, no modelling, no leakage-sensitive transforms.**
Milestone 2 produces one canonical, validated table of historical matches.
