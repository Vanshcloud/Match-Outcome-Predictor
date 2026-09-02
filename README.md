# Match Outcome Predictor

Calibrated home / draw / away probabilities for roughly 38 professional
football competitions, from ingestion through to a served API and dashboard.

[![CI](https://github.com/Vanshcloud/Match-Outcome-Predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/Vanshcloud/Match-Outcome-Predictor/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.13-blue)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

> **Status: Milestone 1 of 13 — repository foundation.**
> The configuration, logging, path and HTTP layers are built and tested.
> Ingestion begins at Milestone 2. Nothing in this repository predicts a match
> yet, and this README will not claim otherwise until it does.

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

Leakage prevention is not a review step here. Milestone 6 is a dedicated
temporal-integrity suite: any feature that can see its own match fails CI.

## Data

**Source: [football-data.co.uk](https://www.football-data.co.uk/)** — free,
static CSV, no API key, no account, no rate limit, no scraping. Two schemas
behind one canonical interface:

| Schema | Competitions | Depth | Per-match detail |
|---|---|---|---|
| Main (`/mmz4281/{season}/{div}.csv`) | 22 divisions across England, Scotland, Germany, Italy, Spain, France, Netherlands, Belgium, Portugal, Turkey, Greece | 1993/94 → present | Result, half-time score, shots, shots on target, corners, fouls, cards, referee, odds |
| Extra (`/new/{COUNTRY}.csv`) | 16 countries: Argentina, Austria, Brazil, China, Denmark, Finland, Ireland, Japan, Mexico, Norway, Poland, Romania, Russia, Sweden, Switzerland, USA | ~2012 → present | Result and closing odds only |

Roughly 250,000 matches. The two schemas differ deliberately in what they
carry, and that difference is modelled rather than hidden: features declare the
data they require, so a competition without shot statistics yields nulls for
those features instead of blocking the pipeline.

Ingestion is an **adapter layer**. Adding API-Football, Understat, StatsBomb or
Football-Data API means writing one adapter against the canonical match schema,
with no change to any downstream code.

> **Not implemented, by decision, not oversight:** Transfermarkt. Its Terms of
> Use (§11.1) prohibit both automated access and training models on its
> content. The registry has a slot for it; the adapter will stay unwritten.

### Features that are not available, and why that is stated up front

Weather, altitude, travel distance, injuries, suspensions, lineups, formation,
manager changes, attendance, squad market value and player ratings are not
available at match level across 38 competitions from any free source. They are
not in the roadmap as "coming soon". The feature registry has slots for them,
so any that later becomes available is one registry entry rather than a
refactor — but the model is built from what genuinely exists: ratings, form,
home advantage, goal difference, rolling shot statistics, rest days, fixture
congestion and head-to-head history.

## Architecture

```
src/
  utils/            paths, typed config, logging, HTTP   [Milestone 1] ✅
  ingestion/        provider adapters -> canonical schema [Milestone 2]
  storage/          DuckDB + Parquet, Postgres for serving [Milestone 3]
  ratings/          Elo and Dixon-Coles, strictly causal   [Milestone 4]
  feature_engineering/  registry, rolling windows          [Milestone 5]
  validation/       schema checks and leakage gates        [Milestone 6]
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
make test      # unit tests
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
| 2 | Ingestion — adapters, canonical schema, league registry | |
| 3 | Storage — DuckDB, Parquet, Postgres, schema validation | |
| 4 | Ratings — Elo, Dixon-Coles | |
| 5 | Feature engineering — registry, rolling windows | |
| 6 | Leakage suite — temporal integrity enforced in CI | |
| 7 | Splits and baselines — walk-forward CV, RPS/log loss | |
| 8 | Model zoo — LR, RF, XGBoost, LightGBM, CatBoost, MLP, Optuna, MLflow | |
| 9 | Ensembling and calibration | |
| 10 | Evaluation and explainability — backtests, SHAP, model cards | |
| 11 | API — FastAPI | |
| 12 | Dashboard — Streamlit + Plotly | |
| 13 | MLOps — Docker, CI/CD, retraining, drift monitoring, deployment | |
| 14 | Research models — TabNet, FT-Transformer, AutoML, benchmarked against the best GBDT with a written verdict | |

## Licence

MIT. See [LICENSE](LICENSE).

Match data is sourced from football-data.co.uk and remains subject to its
terms. No data is redistributed by this repository.
