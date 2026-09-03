# Match Outcome Predictor

Calibrated home / draw / away probabilities for roughly 38 professional
football competitions, from ingestion through to a served API and dashboard.

[![CI](https://github.com/Vanshcloud/Match-Outcome-Predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/Vanshcloud/Match-Outcome-Predictor/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.13-blue)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

> **Status: Milestone 8 of 14 — the model zoo.**
> **303,517 matches** across 39 competitions, 27 countries and 33 years reduce
> to one canonical schema, queryable through a storage interface and checked by
> **24 validation rules** on every ingest. Two ratings now run over it, and the
> stronger one — Dixon-Coles — already reaches **0.9774 log loss** on the
> Premier League against **0.9619** for the bookmaker's closing line, on the
> same 8,818 matches. **Twenty features** join them, every one proved causal by
> recomputation rather than by review. Splits and the model zoo are next.

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

Leakage prevention is not a review step here. Four probes in
`src/validation/temporal.py` test a derived column by recomputing it — truncate
the history and the surviving rows must not move; rewrite one scoreline and
that match's own row must not move; rewrite one input column and see which
outputs move; and no training row may be dated at or after any evaluation row.
They are generic over `Callable[[DataFrame], DataFrame]`, they run on every
build and on every test run over every producer in the codebase, and each is
verified against a planted leak of the kind it exists to find.

## Data

**Source: [football-data.co.uk](https://www.football-data.co.uk/)** — free,
static CSV, no API key, no account, no rate limit, no scraping. Two schemas
behind one canonical interface:

| Schema | Competitions | Depth | Per-match detail |
|---|---|---|---|
| Main (`/mmz4281/{season}/{div}.csv`) | 22 divisions across England, Scotland, Germany, Italy, Spain, France, Netherlands, Belgium, Portugal, Turkey, Greece | 1993/94 → present | Result, half-time score, shots, shots on target, corners, fouls, cards, referee, odds |
| Extra (`/new/{COUNTRY}.csv`) | 17 competitions in 16 country files: Argentina (league + cup), Austria, Brazil, China, Denmark, Finland, Ireland, Japan, Mexico, Norway, Poland, Romania, Russia, Sweden, Switzerland, USA | ~2012 → present | Result and closing odds only |

**39 competitions, 27 countries, 303,517 matches, 1,323 teams, 1993-2026.**
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
in, in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)** — sixteen of them,
including a missing file that returns HTTP 300 with an HTML body, a header
narrower than its own rows, a country file that mixes a league with a cup, and
five files that are copies of another division served under its name.

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

## Features

Twenty columns — form, venue form, rest, congestion, head-to-head — each a
window over what happened before the match. Three mechanisms keep them honest,
and none of them relies on anyone reading the code correctly:

1. **Windows cut on the date, not the row.** `shift(1).rolling(k)` lets two
   matches on the same date inform each other, ordered only by an arbitrary
   tiebreak. Every window here ends at the last row strictly earlier *by date*.
2. **The registry says which features have anything to prove.** Each declares
   the columns it reads; whether it can leak follows from that rather than from
   a field somebody might set wrongly. Seven of the twenty read nothing but who
   is playing and when.
3. **The probes recompute it** — truncate the history, rewrite a scoreline —
   on every build, and the build exits non-zero if either fails.

They carry real signal. Home win rate moves from **31.7% to 61.1%** across the
form-gap range, and the draw rate peaks between evenly matched sides and falls
at both extremes, which is what football says should happen and is not
something the feature was built to produce. Rest days, honestly, carry none at
all — two points of spread and not even monotone. It is kept for the
interaction and will earn its place in Milestone 8's ablation or be dropped.

**[docs/FEATURES.md](docs/FEATURES.md)** has every number, including that one.

## Leakage

The three mechanisms above are per-pipeline, and a pipeline probes the
producers on *its own list*. A list is a thing you can forget to add to. So the
suite does not take one: it walks `src.ratings` and `src.feature_engineering`,
finds every class satisfying the producer contract, and probes each with the
defaults the pipelines use. A producer that exists is a producer that gets
probed, and the mirror check fails if a discovered producer is in no pipeline's
defaults.

Two probes join the two from Milestone 4. **Split boundary**: no training row
may be dated at or after any evaluation row, and ties fail, because a model
trained on the 3pm results is not entitled to predict the 5.30 kick-off.
**Observed reads**: rewrite one input column, recompute, and whatever moved
read it — the measured counterpart to the registry's hand-written declaration,
which is the one part of it that can be wrong silently.

That trace says four things reading the code does not. Elo reads `season` and
never `date`, so it is invariant to *when* matches were played. Dixon-Coles
reads `competition_id` and Elo does not — Elo's per-country pooling is already
inside the team id. `home_venue_points_5` does not read `away_team_id`, which
is the asymmetry a reshape is most likely to get wrong, visible here as an
absence rather than as a number to check by eye. And nothing anywhere reads an
odds column, which is enforced rather than remembered.

**[docs/LEAKAGE.md](docs/LEAKAGE.md)** traces all thirty derived columns back to
the canonical columns they read, and says what the trace cannot tell you.

## Splits and baselines

Walk-forward: five folds of a year each, expanding training window, anchored at
the end of the history. The cut is a **date**, so a full Saturday programme
never lands on both sides of it, and every fold is checked by the split-boundary
probe before it is used.

| 59,001 matches every forecaster could price | log loss | RPS | accuracy |
|---|---:|---:|---:|
| Bookmaker closing odds, overround removed | **0.9993** | **0.2031** | 50.6% |
| Dixon-Coles | 1.0277 | 0.2114 | 48.5% |
| Class prior, counted per fold | 1.0751 | 0.2284 | 43.7% |
| Home always | ∞ | 0.4316 | 43.7% |

Two subsets, always: each forecaster over what *it* could price, and all of
them over what *every* one could price. The bookmaker quotes 99.8% of these
matches and Dixon-Coles prices 95.3%, and putting one number from each beside
the other compares two different questions.

Home-always scores infinite log loss and it is not clipped — a forecast that
ruled out what happened was infinitely wrong. RPS still ranks it, because RPS
is bounded and knows H, D and A are ordered. That contrast is why both are
reported.

Two findings came out of it. The rating's edge over the prior tracks the spread
of team strength in a competition at **r = 0.90** — which is what a strength
model should do and nobody told it to. And the gap to the closing line tracks
that same spread at **−0.14**, meaning what the bookmaker knows on top of the
rating is not strength, and is worth about the same amount everywhere. That is
the target for Milestone 8.

**[docs/EVALUATION.md](docs/EVALUATION.md)** has the fold table, the
per-competition breakdown, and the one competition where Dixon-Coles loses to
counting base rates.

## The model zoo

Six families over thirty columns — the twenty features and the ten rating
columns — scored by the *same* backtest the baselines are, on the same folds
and the same two subsets. A trained model is a forecaster that fits inside its
own `forecast(train, evaluate)`, so nothing in the evaluation layer knows an
estimator exists, and a model and a baseline can be put in one table.

| 59,001 matches every forecaster could price | log loss | RPS | accuracy |
|---|---:|---:|---:|
| Bookmaker closing odds | **0.9993** | **0.2031** | 50.6% |
| CatBoost | **1.0159** | 0.2083 | 49.3% |
| XGBoost / LightGBM / logistic regression | 1.0161–1.0162 | 0.2083 | 49.3% |
| Random forest | 1.0172 | 0.2087 | 49.2% |
| MLP | 1.0203 | 0.2091 | 49.1% |
| Dixon-Coles | 1.0277 | 0.2114 | 48.5% |
| Class prior | 1.0751 | 0.2284 | 43.7% |

Three findings worth stating plainly.

**The top four are within 0.0003.** Logistic regression on thirty columns is
not distinguishable from three tuned gradient-boosting libraries. Milestone 7
found that what the bookmaker knows on top of a strength rating is not
strength; this adds that it is not a non-linear function of these thirty
columns either. What is left is missing information, not missing capacity.

**Form is worth more than either rating.** Withholding the fourteen form
columns costs 0.0033 of log loss; withholding Elo costs 0.0021 and Dixon-Coles
0.0016. Rest days and congestion are worth 0.0003 — Milestone 5 shipped them
saying the ablation would settle it, and it has. Head-to-head is worth 0.0001.

**The zoo fixes the competition the rating could not price.** Dixon-Coles lost
to counting base rates on the Argentine cup, 1.1329 to 1.0902. LightGBM gets
1.0806 there — without a special case, a per-competition rule, or anyone
telling it which competition was awkward.

Tuning bought very little: four of the six searches moved the fourth decimal
place, and twenty trials could not beat `C=1.0` for logistic regression at all.
Every search ran on matches strictly earlier than the first reported fold.

**[docs/MODELS.md](docs/MODELS.md)** has the search budgets, the full ablation,
and what is deliberately not in this milestone.

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

The suite earned its keep immediately. It found one row in 303,517 filed under
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
  validation/       24 + 9 + 10 checks, as one report      [Milestone 3] ✅
    report.py         Check, three outcomes, two severities
    matches.py        the suite: schema, integrity, referential, distribution
    card.py           the generated dataset card
  feature_engineering/  windows over the past             [Milestone 5] ✅
    windows.py        the causal primitive: cut on date, not on row
    registry.py       what each feature reads, and so what it must prove
    team_history.py   form, venue form, rest, congestion
    head_to_head.py   prior meetings
  ratings/          Elo and Dixon-Coles, strictly causal  [Milestone 4] ✅
    base.py           the RatingModel protocol and schema
    elo.py            one pool per country, online updates
    dixon_coles.py    bivariate Poisson, refitted per competition
  validation/temporal.py  four probes: prefix, outcome, split, reads   ✅
  validation/leakage.py   every producer, found rather than listed [Milestone 6] ✅
  models/           splits, baselines, and the zoo         [Milestone 7-8] ✅
    splits.py         walk-forward folds, cut on the date, probed
    baselines.py      home-always, class prior, Dixon-Coles, the closing line
    dataset.py        the thirty columns, derived from the two registries
    zoo.py            six families, one wrapper, a fresh fit per fold
    tuning.py         Optuna, on matches earlier than every reported fold
    tracking.py       MLflow to a local SQLite file, failure-tolerant
    (calibration)                                       [Milestone 9]
  evaluation/       log loss, RPS, accuracy                [Milestone 7] ✅
    metrics.py        two proper scoring rules and one improper one
    (backtests, model cards)                            [Milestone 10]
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
make ratings   # build Elo + Dixon-Coles (~10 min); make ratings-elo is seconds
make features  # build the 20-feature table (~10 seconds)
make audit     # probe every producer, trace every column (~3 min)
make backtest  # score every baseline over walk-forward folds (~5 seconds)
make train     # fit the six model families over the folds (~8 minutes)
make ablation  # what each feature block is worth (~5 minutes)
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
| 5 | Feature engineering — registry, causal windows, 20 features | ✅ |
| 6 | Leakage suite — every producer found and probed, every column traced | ✅ |
| 7 | Splits and baselines — walk-forward CV, RPS/log loss | ✅ |
| 8 | Model zoo — LR, RF, XGBoost, LightGBM, CatBoost, MLP, Optuna, MLflow | ✅ |
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
