# Match Outcome Predictor

Calibrated home / draw / away probabilities for football's top leagues,
benchmarked against the bookmaker's closing line. The dashboard covers 10
competitions — the Premier League, Championship, La Liga, Serie A, Bundesliga,
Ligue 1, Eredivisie, Primeira Liga and Brazil's Série A with forecasts, and the
Champions League with fixtures and live scores. The model learns from the match
history of 39 competitions, and the project runs from raw results to a served
API and that dashboard.

[![CI](https://github.com/Vanshcloud/Match-Outcome-Predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/Vanshcloud/Match-Outcome-Predictor/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.13-blue)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

| | Measured | |
|---|---:|---|
| Out-of-sample forecasts | **62,036** | five expanding walk-forward folds |
| Log loss, shipped blend | **1.0156** | on the 59,001 matches every forecaster could price |
| Log loss, closing line | **0.9993** | the benchmark — the model is behind it, everywhere |
| Pooled calibration error | **0.0015** | over 186,108 probability statements |

## What it does

- Ingests **303,517 matches** (39 competitions, 27 countries, 1,323 clubs,
  July 1993 to September 2026) into one validated schema.
- Builds two causal team ratings and twenty pre-match features, and fits six
  model families on walk-forward folds.
- Serves a calibrated blend over HTTP.
- Shows the forecasts in a Streamlit dashboard for the 10 competitions the
  live feed covers, beside how reliable probabilities of that size have been
  and what the closing line said.

The problem it addresses: football predictions are usually reported as a hit
rate, which hides whether the stated probabilities can be trusted, and are
easy to inflate with information that would not have been available before
kick-off. This project forecasts only from pre-match information, checks that
by recomputation, and scores itself against the market.

It is a probabilistic forecaster, not a tipster. The output is a distribution,
and the project measures whether that distribution is honest: a stated 60%
should happen about 60% of the time. That property matters more than
accuracy.

## Key capabilities

| | |
|---|---|
| Data | Incremental, checksummed ingest of football-data.co.uk CSVs; a 35-column canonical schema; 24 validation checks on every ingest; a generated dataset card |
| Features | Elo and Dixon-Coles ratings plus 20 form, schedule and head-to-head features, each a window over matches strictly before kick-off |
| Leakage control | Recomputation probes over every feature and rating producer, and a split-boundary probe on every fold; both run in CI |
| Models | Logistic regression, random forest, XGBoost, LightGBM, CatBoost and an MLP; a blend chosen on error correlation; temperature scaling |
| Evaluation | Log loss, ranked probability score (RPS) and accuracy on five walk-forward folds, against four baselines including the de-vigged closing line; reliability tables and a generated model card |
| Serving | FastAPI service with checksum-verified model artefacts, an `in_sample` flag on every response, an LRU cache, Prometheus metrics and an optional PostgreSQL prediction log |
| Dashboard | Home, live centre, competitions, match detail, search and model pages for the 10 competitions on football-data.org's free plan; saved favourites; live scores, fixtures and club crests; webhook notifications |
| Closing the loop | Upcoming fixtures are priced before kick-off, logged, and scored once their results are ingested |

## Screenshots

Captured from the running dashboard, with the data snapshot described under
Reproducibility and the live football-data.org feed connected.

![Home page: live and upcoming matches](docs/screenshots/home.png)
*Home page: matches in play and the next seven days across the followed
competitions. Scores and club crests are served at runtime by football-data.org;
the crests belong to their clubs and none are stored in this repository.*

![Match page for a played fixture](docs/screenshots/match.png)
*Match-level forecast with H/D/A probabilities, market comparison and
evaluation-status context. This match is inside the served model's training
window, and the page says so.*

![Model page](docs/screenshots/model.png)
*Model evaluation and calibration view: the headline numbers, every forecaster
on the same matches, and tabs for reliability, per-competition results and
limitations.*

Without data, the dashboard still starts, and every section says what is
missing and which command produces it.

## Architecture

```
football-data.co.uk CSVs ──▶ src/ingestion ──▶ matches.parquet (35 cols) ──▶ src/validation
                                                    │
                                                    ▼
                         src/ratings (Elo, Dixon-Coles) + src/feature_engineering (20 features)
                                                    │   probed by src/validation/temporal + leakage
                                                    ▼
                       30 design columns ──▶ src/models (walk-forward folds, zoo, blend, calibration)
                                                    │
                           ┌────────────────────────┴─────────────────────────┐
                           ▼                                                  ▼
               src/evaluation → reports, model card          make model → models/servable.joblib
                                                                              │
          fixtures.csv ──▶ make fixtures ──▶ upcoming design rows ──▶ api/ (FastAPI) ──▶ PostgreSQL log
                                                                              │              │
                                             dashboard/ (Streamlit, HTTP client only) ◀──────┘  make archive
                                             + football-data.org live fixtures
```

- **One direction of dependency, enforced.** `api` imports `src`, and nothing
  in `src` imports `api`. The dashboard talks to the API over HTTP and never
  imports it. The dashboard itself is layered
  `views → services → providers → domain`, and no view names a concrete
  provider. Eighteen architectural invariants in CI hold this shape
  (`make invariants`).
- **No second implementation of the feature layer.** The API looks up design
  rows written by the batch build and never computes features inside a request.
  Upcoming fixtures get their rows from the same builders, run over the
  history with the fixtures appended.

## ML methodology

1. **Ratings.**
   - *Elo* uses one pool per country, so a promoted club keeps its history. It
     is updated online, so a match's rating depends only on earlier matches.
     Its constants were fitted on matches before 2005-07-01.
   - *Dixon-Coles* is a bivariate Poisson with a low-score correction and time
     decay. It is refitted per competition on a window that ends strictly
     before the refit date.
2. **Features.** Twenty columns: form, venue form, goals and shots over the
   last five matches, rest days, fixture congestion, and head-to-head. Windows
   cut on the **date**, not the row, so two matches on the same day never
   inform each other. Nulls mean "no history yet" and are kept as nulls.
3. **Models.** Six families on the thirty design columns. Hyperparameters were
   tuned with Optuna on a *tuning slice* that ends before the first reported
   fold.
4. **Blend.** The unweighted mean of XGBoost, logistic regression and the MLP.
   Members are admitted best-first when their per-match errors correlate below
   0.99, on the tuning slice, with every member already in. The other three tree
   models correlate at 0.993–0.996 with XGBoost, which goes in first, so only
   one tree is kept.
5. **Calibration.** One temperature, `p^(1/T)`. It is fitted on the last 365
   days of each fold's training half, with the model refitted on everything
   before that.

Odds are **not** a model input. They are the benchmark, and a test fails if any
feature reads them.

## Dataset and temporal split

- **Source.** [football-data.co.uk](https://www.football-data.co.uk/): 22 main
  divisions, most starting between 1993/94 and 1997/98, plus 17 extra-schema
  competitions, most starting in 2012. The extra schema has results and odds
  only, with no shot statistics.
- **Validation.** Every ingest runs 24 checks: schema, integrity, referential
  and distribution. The real table has no duplicate ids, no duplicate
  fixtures, and no result that disagrees with its score.
- **Split.** Five **expanding walk-forward folds** of 365 days each, anchored at
  the latest match. Every fold trains only on matches strictly before its first
  evaluation day:

| Fold | Trains to | Scores (end exclusive) | Matches |
|---|---|---|---:|
| 0 | 2021-09-02 | 2021-09-03 → 2022-09-03 | 13,030 |
| 1 | 2022-09-02 | 2022-09-03 → 2023-09-03 | 12,302 |
| 2 | 2023-09-02 | 2023-09-03 → 2024-09-02 | 12,503 |
| 3 | 2024-09-01 | 2024-09-02 → 2025-09-02 | 12,399 |
| 4 | 2025-09-01 | 2025-09-02 → 2026-09-02 | 11,802 |

That is 62,036 out-of-sample forecasts. The bookmaker and Dixon-Coles can both
price **59,001** of them, and every comparison below uses that shared subset.

## Leakage prevention

- **Date-cut windows.** Every window ends at the last match strictly earlier
  *by date*.
- **A feature registry.** Each feature declares the canonical columns it reads,
  so which features can leak is derived rather than asserted.
- **Four recomputation probes** (`src/validation/temporal.py`):
  - Truncate the history, and the surviving rows must not change.
  - Rewrite one scoreline, and that match's own row must not change.
  - Rewrite one input column, and whatever moves must be declared as reading
    it.
  - No training row may be dated on or after any evaluation row.
- **Found, not listed.** The leakage suite walks the packages to find every
  producer and probes each one. It also checks that planted leaks are caught: a
  shuffled split, a split through a matchday, and a feature reading the odds.
- **Everything fitted sees only training data.** Imputation and scaling are fit
  inside each fold's pipeline. Tuning, member selection and calibration all use
  data before the evaluation half.

One stated caveat: some rating settings were chosen by looking at data that
overlaps the reported folds. These are the Dixon-Coles decay, window and refit
interval (chosen on the Premier League), and which Elo refinements to keep.
They are a handful of scalars on a flat response surface.
[docs/LEAKAGE.md](docs/LEAKAGE.md) traces all thirty columns.

## Evaluation metrics

**Log loss** and **RPS** are the primary metrics. RPS respects the ordering
H > D > A, so predicting an away win when the home side won costs more than
predicting a draw. **Accuracy** is reported last: draws are about a quarter of
matches and are almost never the most likely single outcome, so a
well-calibrated model rarely predicts one. **Reliability** is measured, not
assumed.

## Results

On the 59,001 matches every forecaster could price:

| Forecaster | Log loss | RPS | Accuracy |
|---|---:|---:|---:|
| Bookmaker closing odds, overround removed | **0.9993** | **0.2031** | 50.6% |
| **Blend of three, calibrated (shipped)** | **1.0156** | **0.2082** | 49.3% |
| CatBoost | 1.0159 | 0.2083 | 49.2% |
| XGBoost / LightGBM / logistic regression | 1.0161–1.0162 | 0.2083–0.2084 | 49.2–49.3% |
| Random forest | 1.0172 | 0.2087 | 49.2% |
| MLP | 1.0203 | 0.2091 | 49.1% |
| Dixon-Coles rating alone | 1.0277 | 0.2114 | 48.5% |
| Class prior, counted per fold | 1.0751 | 0.2284 | 43.7% |

- **The model is behind the market in all 39 competitions.** The blend trails
  the closing line by **0.0163 ± 0.0007** of log loss (standard error of the
  per-match difference). A model that beat the closing line on public data
  would more likely have a leak than an edge.
- **The top models are nearly level.** The blend is 0.0005 better than
  XGBoost, its best member, and 0.0003 better than CatBoost. Paired over the
  193 fold × competition cells, those gaps are about 2.4 and 1.4 standard
  errors. So the blend is measurably better than the models it averages, and
  level with CatBoost.
- **The gap to the market is concentrated.** Where the model is within 2% of
  the closing line (9,628 forecasts) it is level with it (+0.0009). Where it is
  more than 20% away (829 forecasts) it loses by 0.18. A large disagreement
  signals model error, not value, so the dashboard offers no betting signal.

Details:
- [docs/EVALUATION.md](docs/EVALUATION.md): baselines, folds, per-competition
  results, the market bands.
- [docs/MODELS.md](docs/MODELS.md): the zoo, tuning, ablation, the blend.
- [docs/EXPLAINABILITY.md](docs/EXPLAINABILITY.md): SHAP, permutation and
  ablation by feature block.

## Calibration and reliability

**Pooled calibration error: 0.0015**, over 186,108 out-of-sample statements
(62,036 matches × 3 outcomes, 10 bins). A few bins:

| Stated | Statements | Mean stated | Happened |
|---|---:|---:|---:|
| 20–30% | 78,365 | 26.2% | 26.2% |
| 50–60% | 13,565 | 54.4% | 54.7% |
| 60–70% | 6,092 | 64.2% | 64.7% |
| 80–90% | 1,071 | 84.0% | 82.3% |

Reliability is not uniform:
- The rare 80%+ statements are somewhat overconfident.
- Per competition, the calibration error runs up to 0.027 (the Argentine cup).
- The generated [model card](docs/MODEL_CARD.md) lists the least reliable
  competitions and what the model must not be used for, including betting.

## Dashboard and live data

The dashboard is a **client** of the service. It reads results from the match
table, probabilities from the API over HTTP, and reports from disk. When a
source is missing, each section names that source and the command that
produces it; nothing invents a fixture.

| Page | Contents |
|---|---|
| Home / Live centre | Live matches, and the coming week's fixtures each with the model's probabilities; auto-refresh every 60 s while open |
| Competitions | The nine leagues the live feed covers plus the Champions League, as a flag list; each page lists the coming week's fixtures with their probabilities |
| Match | Opened from any card, not the sidebar: the forecast, and how often forecasts in that probability band came true; form and head-to-head for a match already played |
| Search | A club in the covered leagues, with its crest, its live and upcoming fixtures, and its results |
| Model | Headline numbers; every forecaster on the same matches; a reliability diagram filterable by competition and year; per-competition reliability; the served-forecast archive; limitations |

- **Live and upcoming fixtures** come from the optional
  [football-data.org](https://www.football-data.org/) provider. It covers 10
  competitions on the free tier — 9 leagues with forecasts and the Champions
  League without — and is polled at most once a minute; there is
  no push feed and no minute-by-minute data.
- **Notifications.** When a tab is open, goals, kick-offs and full-times in the
  matches you follow appear in the page and can also be posted to a webhook.
  There is no background watcher.
- **Forecasts for played matches are in-sample.** The served model is fitted on
  the whole history, so a forecast for a match that has already been played
  comes from a model that saw its result. The API flags these, and the match
  page says so. Out-of-sample served forecasts come from `make fixtures`
  followed by `make price`.
- **Scoring the served forecasts.** `make archive` scores the logged forecasts
  once their results have been ingested, and reports how large a shift the
  archive could detect. Per-match log loss has a standard deviation of 0.3976,
  so detecting a 0.0163 shift takes **2,286** scored forecasts. No
  out-of-sample served forecast has been scored yet: a forecast becomes
  scorable only after its match is played and `make data` ingests the result.

## Quick start

Requires Python 3.13. On macOS, LightGBM and XGBoost also need
`brew install libomp`.

```bash
make setup      # venv, dev dependencies, git hooks
make test       # unit suite — no data, no network needed
```

Build the data and a servable model (about 30 minutes, mostly download and
ratings):

```bash
make data       # download and ingest all 39 competitions
make ratings    # Elo + Dixon-Coles
make features   # the 20-feature table
make model      # fit the shipped blend on the whole history → models/
```

Serve it:

```bash
make api        # http://127.0.0.1:8000/docs
make dashboard  # http://127.0.0.1:8501 (bound to localhost); reads the API
```

To rebuild the evaluation reports and the model card as well, use
`make reproduce` (about 60 minutes, and it runs `make setup` first). Or run the
steps separately: `make train`, `make ensemble`, `make explain`, `make card`.

To price upcoming fixtures and score them later:

```bash
make fixtures   # design rows for the published fixture list (~12 min)
# restart the API so it indexes them, then:
make price      # ask the service about each; logged if PREDICTION_LOG_DSN is set
make data && make archive   # after the matches are played
```

Docker: `make model` first, then `docker compose up`. This starts the API, the
dashboard and PostgreSQL, with `data/` and `models/` mounted read-only. The
compose file uses a development database password and is **not** a production
deployment. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

`make help` lists every target.

## Configuration

Settings come from `configs/config.yaml` (unknown keys are rejected), overlaid
by environment variables. Every variable is optional. See
[.env.example](.env.example).

| Variable | Purpose |
|---|---|
| `DATA_DIR`, `MODEL_DIR` | Where tables and the artefact live |
| `API_PORT`, `API_MAX_BATCH`, `API_PREDICTION_CACHE` | Service port, batch limit, cache size |
| `PREDICTION_LOG_DSN` | PostgreSQL prediction log (secret; unset disables the log) |
| `DASHBOARD_API_URL` | Where the dashboard asks for forecasts |
| `DASHBOARD_FIXTURE_PROVIDER` | `none` (default) or `football-data.org` |
| `FOOTBALL_DATA_API_KEY` | football-data.org key (secret) |
| `DASHBOARD_NOTIFIER`, `DASHBOARD_WEBHOOK_URL` | `none` or `webhook`, plus its URL (secret) |
| `DASHBOARD_PROFILE_STORE` | Where saved favourites are written |
| `LOG_LEVEL` | Logging level |

Optional sign-in uses Streamlit's OIDC support, configured in
`.streamlit/secrets.toml` (gitignored). It also needs
`pip install "streamlit[auth]"`. See [docs/DASHBOARD.md](docs/DASHBOARD.md).

## Setting up football-data.org

The historical pipeline needs no key. The live fixtures and scores on the
dashboard, and the week of fixtures `make fixtures` prices, do:

1. Register for a free key at
   [football-data.org/client/register](https://www.football-data.org/client/register).
2. Set the variables:
   ```bash
   export FOOTBALL_DATA_API_KEY=...
   export DASHBOARD_FIXTURE_PROVIDER=football-data.org
   ```
3. Covered competitions on the free plan: Premier League, Championship,
   Bundesliga, Serie A, La Liga, Ligue 1, Eredivisie, Primeira Liga, Brazil
   Série A, and the Champions League (fixtures and scores only: no match
   history for it is ingested, so it has no forecasts).

Requests are split into windows of at most 10 days and cached for 60 s. Retries honour `Retry-After`. If the key is invalid or the feed
is down, the affected section shows the feed's own error message and the rest
of the page carries on.

The historical data (football-data.co.uk) and the live feed
(football-data.org) are two different organisations. Nothing from the live
feed is written to a table or read by the model.

## Testing and quality gates

| Gate | Command | Also in CI |
|---|---|---|
| Unit tests, 100% statement coverage of `src`, `api`, `dashboard` | `make test-cov` | ✓ |
| Integration tests on real data (skip without it) | `make test-int` | — |
| Lint / format | `make lint`, `make format-check` (ruff, black) | ✓ |
| Types | `make typecheck` (mypy, strict) | ✓ |
| Architecture | `make invariants` (18 boundary checks) | ✓ |
| Leakage suite | part of the unit suite | ✓ (own step) |
| Images | build both, start them, assert `/health`, `/metrics`, non-root | ✓ |

Deprecation warnings are errors. The unit suite is offline and uses synthetic
data shaped like the real files.

## Known limitations

- **It does not beat the market,** in any of the 39 competitions. Do not use it
  for betting.
- **Inputs are limited to what free, match-level data carries:** results, shots
  (main divisions only), dates and ratings. There are no lineups, injuries,
  transfers or shot-level xG.
- **A club with no history** gets forecasts close to the base rates.
- **Reliability varies by competition,** and very confident statements (80%+)
  are slightly overconfident.
- **The top model families are nearly indistinguishable,** and the blend's edge
  over CatBoost is within noise.
- **The served archive has no scored out-of-sample forecasts yet.** A drift
  verdict needs about 2,286.
- **Fixture design rows** come only for competitions listed in
  football-data.co.uk's `fixtures.csv`. Building them re-runs the ratings over
  the whole history (about 12 minutes).
- **A club with two fixtures in one build:** the later fixture's form window
  contains the earlier, unplayed one. It averages the known results among its
  last five and treats the missing one like any other result-less row.
- **The dashboard is unauthenticated by default and not meant for public
  hosting.** Named profiles have no password.

## Reproducibility

**No credential reproduces everything in the Results and Calibration sections
above.** The historical pipeline — ingest, ratings, features, training,
evaluation, model card — reads only football-data.co.uk, which is a public
static host and needs no key. Secrets are needed for optional extras alone:
live fixtures, the PostgreSQL prediction log, webhook notifications, sign-in.

| What | Command | Needs | Time |
|---|---|---|---|
| Unit suite (1,675 tests), offline, on a clean checkout | `make setup && make test` | — | ~5 min, plus the install |
| The 39-competition match table | `make data` | ~70 MB download | ~15 min |
| Ratings, features, servable model | `make ratings features model` | the table | ~11 min |
| Every report and the model card | `make reproduce` | the above | ~60 min |
| API + dashboard, locally | `make api`, `make dashboard` | a model | seconds |
| API + dashboard + PostgreSQL | `docker compose up` | Docker, `make model` first | ~5 min build |
| Live fixtures and scores | as above | `FOOTBALL_DATA_API_KEY` (free tier) | — |

Disk after a full build: about 115 MB of data plus 3 MB of model artefacts.

- **Deterministic given the same input data.** Each stage rewrites identical
  bytes from identical inputs: the ingest is checksummed, and re-running the
  baseline backtest reproduces its report file exactly. Dependencies are pinned
  in the `requirements*.txt` files, and `uv.lock` pins the transitive closure.
  Parallel model fits can differ at about 1e-9.
- **The reported numbers describe one data snapshot.** It is the table as
  ingested on 2026-09-03: last match 2026-09-01, 303,517 rows,
  `matches.parquet` sha256 `41ad0914…`.
  - The data is not redistributed, so a fresh clone must run `make data`.
  - `make data` fetches the provider's *current* files. New matches move the
    walk-forward folds, which are anchored at the latest match, so a later
    rebuild produces **different** numbers from the same code. **A fresh clone
    reproduces the method, not the digits**, and the further from 2026-09-03 it
    runs the further they drift.
- **Not reproducible by design:**
  - The prediction archive, which is a log of what was served.
  - The upcoming-fixture table, which is true for about a week.
  - Live dashboard data.

## Security

The optional services use four secrets: `PREDICTION_LOG_DSN`,
`FOOTBALL_DATA_API_KEY`, `DASHBOARD_WEBHOOK_URL`, and the OIDC secrets in
`.streamlit/secrets.toml`. All of them come from the environment or that
gitignored file, never from committed config. Webhook errors never log the
URL. With sign-in enabled, a reader's email is stored locally as the key for
their favourites. See [SECURITY.md](SECURITY.md).

## Project structure

```
src/
  ingestion/            provider adapter, canonical schema, fixture list, manifests
  storage/              DuckDB over Parquet; PostgreSQL prediction log
  validation/           data checks, dataset card, temporal and leakage probes
  ratings/              Elo, Dixon-Coles
  feature_engineering/  feature registry and date-cut windows
  models/               splits, baselines, zoo, tuning, blend, calibration, artefact
  evaluation/           metrics, reliability, market comparison, archive, model card
  explainability/       SHAP and permutation importance
  pipelines/            orchestration for each stage, serving index, fixtures
  utils/                config, logging, paths, HTTP client
api/                    FastAPI service
dashboard/              Streamlit app: views, services, providers, domain
scripts/                the CLIs behind the make targets
configs/                config.yaml, leagues.yaml (the competition registry)
docs/                   design and results documentation
tests/                  unit and integration suites
```

| Document | Covers |
|---|---|
| [DATA_SOURCES](docs/DATA_SOURCES.md), [DATASET_CARD](docs/DATASET_CARD.md) | The provider and its quirks; the generated dataset card |
| [RATINGS](docs/RATINGS.md), [FEATURES](docs/FEATURES.md), [LEAKAGE](docs/LEAKAGE.md) | Ratings, features, leakage defence |
| [EVALUATION](docs/EVALUATION.md), [MODELS](docs/MODELS.md), [EXPLAINABILITY](docs/EXPLAINABILITY.md), [MODEL_CARD](docs/MODEL_CARD.md) | Results, models, attribution, the generated card |
| [API](docs/API.md), [DASHBOARD](docs/DASHBOARD.md), [DEPLOYMENT](docs/DEPLOYMENT.md) | Service, dashboard, operations |
| [DEVELOPMENT_HISTORY](docs/DEVELOPMENT_HISTORY.md), [CHANGELOG](CHANGELOG.md) | How the project was built, milestone by milestone |

## License and attribution

MIT — see [LICENSE](LICENSE).

Match data comes from [football-data.co.uk](https://www.football-data.co.uk/)
and remains subject to its terms. This repository does not redistribute it.
Live fixtures and scores come from
[football-data.org](https://www.football-data.org/) under your own API key.
Transfermarkt is deliberately not used, because its terms prohibit automated
access and model training.
