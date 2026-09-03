# Match Outcome Predictor

Calibrated home / draw / away probabilities for roughly 38 professional
football competitions, from ingestion through to a served API and dashboard.

[![CI](https://github.com/Vanshcloud/Match-Outcome-Predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/Vanshcloud/Match-Outcome-Predictor/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.13-blue)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

> **Status: Milestone 4 of 14 — ratings.**
> **305,499 matches** across 39 competitions, 27 countries and 33 years reduce
> to one canonical schema, queryable through a storage interface and checked by
> **23 validation rules** on every ingest. Two ratings now run over it, and the
> stronger one — Dixon-Coles — already reaches **0.9774 log loss** on the
> Premier League against **0.9619** for the bookmaker's closing line, on the
> same 8,818 matches. Features and the model zoo are next.

---

## What this is honest about

Most football prediction projects report a headline accuracy and stop. This one
states its ceiling first, because the ceiling is the interesting part.

**Bookmaker closing odds — the strongest public benchmark — achieve roughly
0.95–0.97 log loss and 53–54% accuracy on 1X2 markets.** Models built on public
data land around 0.99–1.03 log loss. A football outcome model reporting 70%
accuracy has a data leak, almost without exception; the usual culprit is a
feature computed over a window that includes the match being predicted.

Three consequences shape the whole design:

1. **Log loss and Ranked Probability Score are the primary metrics, not
   accuracy.** RPS is the standard in the forecasting literature because H/D/A
   is *ordinal*: predicting Away when the result was Home is a worse error than
   predicting Draw. Accuracy, F1 and ROC-AUC are reported, but they are not
   what the models are selected on.
2. **Calibration is the deliverable.** A probability of 0.61 should be right
   about 61% of the time. That is the property a probability is *for*, and it
   is measured here with reliability curves, not asserted.
3. **Draws are close to unpredictable.** They occur in roughly a quarter of
   matches and are almost never the most likely single outcome. A
   well-calibrated model that rarely *predicts* "draw" is behaving correctly,
   not failing — which is precisely why accuracy is the wrong target.

Leakage prevention is not a review step here. Two probes in
`src/validation/temporal.py` test any derived column by recomputing it —
truncate the history and the surviving rows must not move; rewrite one
scoreline and that match's own row must not move. They are generic over
`Callable[[DataFrame], DataFrame]`, they run on every ratings build, and they
are verified against planted leaks of both kinds. Milestone 6 points them at
the feature layer.

## Data

**Source: [football-data.co.uk](https://www.football-data.co.uk/)** — free,
static CSV, no API key, no account, no rate limit, no scraping. Two schemas
behind one canonical interface:

| Schema | Competitions | Depth | Per-match detail |
|---|---|---|---|
| Main (`/mmz4281/{season}/{div}.csv`) | 22 divisions across England, Scotland, Germany, Italy, Spain, France, Netherlands, Belgium, Portugal, Turkey, Greece | 1993/94 → present | Result, half-time score, shots, shots on target, corners, fouls, cards, referee, odds |
| Extra (`/new/{COUNTRY}.csv`) | 17 competitions in 16 country files: Argentina (league + cup), Austria, Brazil, China, Denmark, Finland, Ireland, Japan, Mexico, Norway, Poland, Romania, Russia, Sweden, Switzerland, USA | ~2012 → present | Result and closing odds only |

**39 competitions, 27 countries, 305,499 matches, 1,363 teams, 1993-2026.**
Those are measured from a real full ingest, not estimated.
The two schemas differ deliberately in what
they carry, and that difference is modelled rather than hidden: each
competition declares its `Capability` set, so one without shot statistics
yields nulls for those columns instead of blocking the pipeline.

Adding a league is a `configs/leagues.yaml` entry — no code change. Adding a
*provider* means writing one adapter against the canonical schema, with no
change to any downstream code.

Re-running is **incremental and idempotent**. Change detection uses HTTP
conditional requests — the provider answers `If-None-Match` with a 304 and no
body — rather than a file-age guess. Measured over the full ingest: the second
run transfers **0 files and 0.00 MB** and produces a **byte-identical** Parquet
(`sha256 8d489ed3…`). Every cached file is checksummed into
`data/raw/manifest.json`, so which bytes produced a given table is checkable.

**Every quirk that shaped this design is documented, with the file it was found
in, in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)** — fifteen of them,
including a missing file that returns HTTP 300 with an HTML body, a header
narrower than its own rows, and a country file that mixes a league with a cup.

> **No data is redistributed by this repository, and none is committed** — not
> even a test slice. The provider publishes no licence granting redistribution.
> The unit suite runs on synthetic fixtures shaped like the real files, so it
> needs no network; tests requiring real data are marked `integration` and skip
> when it is absent. Run `make data` to fetch your own copy.

> **Not implemented, by decision, not oversight:** Transfermarkt. Its Terms of
> Use (§11.1) prohibit both automated access and training models on its
> content. The registry has a slot for it; the adapter will stay unwritten.

### Features that are not available, and why that is stated up front

Weather, altitude, travel distance, injuries, suspensions, lineups, formation,
manager changes, attendance, squad market value and player ratings are not
available at match level across 39 competitions from any free source. They are
not in the roadmap as "coming soon". The feature registry has slots for them,
so any that later becomes available is one registry entry rather than a
refactor — but the model is built from what genuinely exists: ratings, form,
home advantage, goal difference, rolling shot statistics, rest days, fixture
congestion and head-to-head history.

## Ratings

Two, and neither can see the match it is rating.

**Elo**, one pool per country so a promoted club keeps its history. Online, so
the rating carried into match *n* depends on matches 1..*n*-1 by construction —
no window to get wrong, no whole-table statistic to include by accident. Home
advantage is worth 3.9% of its error; the margin-of-victory multiplier,
autocorrelation damping and season carry-over are worth a few parts in a
thousand each.

**Dixon-Coles**, a bivariate Poisson fitted per competition on a rolling
window that ends *strictly before* the match that triggered the refit. A full
Saturday programme is one round, and a model fitted on the 3pm results to
predict the 5.30 kick-off would look excellent and be useless. It emits a real
H/D/A distribution, which is what makes the table below possible:

| Premier League, the 8,818 matches all three can price | log loss | RPS |
|---|---|---|
| Class prior — predict the base rates every time | 1.0636 | 0.2285 |
| **Dixon-Coles** | **0.9774** | **0.1988** |
| Bookmaker closing odds, overround removed | 0.9619 | 0.1941 |

Every constant in both models was measured rather than inherited, and three of
the conventional values lost: the football-Elo season carry-over is too
aggressive, the Dixon-Coles decay half-life is too fast, and the low-score
correction — the model's defining feature — is worth 0.0002 of log loss here.
The per-competition Elo fitting the plan called for was built, measured and
removed for making the ratings worse. **[docs/RATINGS.md](docs/RATINGS.md)** has
every number, including the ones that did not work.

## Storage and validation

The canonical table is read through a `MatchStore`, never by opening a path.
DuckDB implements it as **views over the Parquet** the ingest pipeline writes —
no load step, no second copy, no server. Filters are pushed into the query, so
a point-in-time read (`until="2015-06-30"`) is the interface's own operation
rather than something every caller reimplements against a materialised frame.
CI enforces the seam: nothing outside `src/storage` reads the table directly.

**Twenty-three checks** run on every ingest, over the frame the pipeline just
wrote — schema, integrity, referential and distribution. Each threshold is a
measurement rather than a guess, and the comment on it says what was measured.
They report rather than gate: a table you can inspect beats one the pipeline
refused to save, and severity decides what stops a caller.

The suite earned its keep immediately. It found one row in 305,499 filed under
the wrong season — an Argentinian match played 2015-01-29 and labelled 2013-14
in the provider's own file — which is reported as a warning and left in place.

### Leakage, stated at the schema

Every canonical column declares which side of kick-off it is knowable on, and a
check fails if any column is unclassified. The defence has to be *temporal*
rather than columnar: `home_shots` is a summary of the ninety minutes, but a
team's shots in its *earlier* matches are a perfectly legitimate feature, so it
cannot be enforced by leaving data out. Odds are pre-match and therefore safe —
and are still reserved as the benchmark, because a model trained on them learns
to copy the bookmaker.

**[docs/DATASET_CARD.md](docs/DATASET_CARD.md) is generated from the table** on
every validation run, with per-competition coverage and the checksum of the
exact file it describes. Aggregate coverage hides what matters: 93% of recent
matches in stat-capable competitions carry shot data, and the National League
carries almost none.

## Architecture

```
src/
  utils/            paths, typed config, logging, HTTP   [Milestone 1] ✅
  ingestion/        provider adapters -> canonical schema [Milestone 2] ✅
    base.py           the 35-column canonical schema + MatchProvider protocol
    csv_reader.py     encodings, ragged rows, HTML-served-as-CSV
    registry.py       the competition registry and season labels
    teams.py          canonical team ids
    football_data.py  the adapter
    manifest.py       checksum-based dataset versioning
  storage/          DuckDB views over Parquet             [Milestone 3] ✅
    base.py           the MatchStore protocol
    duckdb_store.py   the analytical store; point-in-time reads
  validation/       23 data-quality checks as a report    [Milestone 3] ✅
    report.py         Check, three outcomes, two severities
    matches.py        the suite: schema, integrity, referential, distribution
    card.py           the generated dataset card
  ratings/          Elo and Dixon-Coles, strictly causal  [Milestone 4] ✅
    base.py           the RatingModel protocol and schema
    elo.py            one pool per country, online updates
    dixon_coles.py    bivariate Poisson, refitted per competition
  validation/temporal.py  prefix invariance, outcome independence  ✅
  feature_engineering/  registry, rolling windows          [Milestone 5]
  models/           splits, zoo, tuning, calibration    [Milestones 7-9]
  evaluation/       backtests, metrics, model cards       [Milestone 10]
  explainability/   SHAP, permutation importance         [Milestone 10]
  pipelines/        the orchestration each stage exposes
api/                FastAPI service                      [Milestone 11]
dashboard/          Streamlit + Plotly                   [Milestone 12]
```

`src/utils` is the bottom of the dependency graph and imports nothing else from
`src`. CI enforces that, because a cycle is far cheaper to prevent than to
unpick.

## Quick start

```bash
make setup     # venv, dev dependencies, git hooks
make leagues   # list the 39 configured competitions
make data      # download and ingest everything (~15 min first time)
make refresh   # incremental re-run: conditional requests only, 0 MB if unchanged
make validate  # run the data checks and regenerate docs/DATASET_CARD.md
make ratings   # build Elo + Dixon-Coles (~20 min); make ratings-elo is seconds
make test      # unit tests — no network, no data needed
make test-int  # integration tests — needs `make data`
make quality   # ruff + black + mypy
```

Requires Python 3.13. From Milestone 8, macOS also needs `brew install libomp`
for LightGBM and XGBoost.

Configuration is `configs/config.yaml`, overlaid by an explicit set of
environment variables (see `.env.example`). Unknown keys are an error, not a
shrug — a misspelt setting fails at load naming the key, rather than appearing
to work forever.

## Engineering standards

| Gate | Tool | Enforced by |
|---|---|---|
| Lint | ruff | `make lint`, CI |
| Format | black | `make format-check`, CI |
| Types | mypy, strict | `make typecheck`, CI |
| Tests | pytest, ≥95% coverage | `make test-cov`, CI |
| Deprecations | `-W error::DeprecationWarning` | pytest config |
| Authorship | `scripts/hooks/commit-msg` | git hook + CI |

Lint tooling is pinned **exactly**. Unpinned, a formatter release turns CI red
with no code change and disagrees with every local run.

`requirements.txt` grows one milestone at a time. A pinned dependency nothing
imports is a supply-chain surface with no upside.

## Roadmap

| # | Milestone | Status |
|---|---|---|
| 1 | Foundation — config, logging, paths, HTTP, gates | ✅ |
| 2 | Ingestion — adapters, canonical schema, league registry | ✅ |
| 3 | Storage and validation — DuckDB over Parquet, 23 checks, dataset card | ✅ |
| 4 | Ratings — Elo, Dixon-Coles, causality probes | ✅ |
| 5 | Feature engineering — registry, rolling windows | |
| 6 | Leakage suite — temporal integrity enforced in CI | |
| 7 | Splits and baselines — walk-forward CV, RPS/log loss | |
| 8 | Model zoo — LR, RF, XGBoost, LightGBM, CatBoost, MLP, Optuna, MLflow | |
| 9 | Ensembling and calibration | |
| 10 | Evaluation and explainability — backtests, SHAP, model cards | |
| 11 | API — FastAPI, PostgreSQL for served predictions | |
| 12 | Dashboard — Streamlit + Plotly | |
| 13 | MLOps — Docker, CI/CD, retraining, drift monitoring, deployment | |
| 14 | Research models — TabNet, FT-Transformer, AutoML, benchmarked against the best GBDT with a written verdict | |

## Licence

MIT. See [LICENSE](LICENSE).

Match data is sourced from football-data.co.uk and remains subject to its
terms. No data is redistributed by this repository.
