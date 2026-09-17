# Changelog

Notable changes, grouped by version. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). No version has
been tagged as a GitHub release yet; `src/__init__.py` reports 0.12.0.

The full per-milestone record, with the measurements and dead ends behind each
decision, is in [docs/history/MILESTONE_LOG.md](docs/history/MILESTONE_LOG.md).

## [Unreleased]

Milestones 13–20 and the publication pass.

### Added

- **Live fixtures** (M13): an optional football-data.org provider for the 10
  competitions on its free plan, cached for 60 s, with the feed's own error
  shown when it refuses a request.
- **Profiles and favourites** (M14): named profiles, optional OIDC sign-in,
  saved leagues and clubs.
- **Live tracking** (M15): a live strip polled once a minute, kick-off / goal /
  full-time events, optional webhook notifications.
- **Operations** (M16): images published to GHCR on a tag, Prometheus metrics,
  a prediction cache.
- **Prediction archive** (M19): served forecasts scored against results, with
  the sample size a drift verdict needs.
- **Pre-kick-off pricing** (M20): `make fixtures` and `make price` build design
  rows for upcoming fixtures and log out-of-sample forecasts.
- `make invariants` runs CI's architectural checks locally.
- Every derived table and report manifest records an `inputs` block: the name
  and SHA-256 of each table it was built from, so a derived table left stale by
  a later ingest is a hash comparison rather than a guess.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, a bug report form and a pull
  request template.
- Tests over the ignore rules themselves: every secret-bearing path is checked
  against `git check-ignore` and against `.dockerignore`, so a renamed or
  reordered rule fails the suite instead of publishing a key.

### Changed

- The dashboard is organised around the live feed: home, live centre,
  competitions, match, search and model pages, with club crests.
- The closing-line panel (M17) and the squad panel (M18) were removed from the
  dashboard. The closing line remains the benchmark on the Model page.
- Competitions that share a name are labelled with their country
  ("Serie A — Italy", "Serie A — Brazil").
- A cold home page sends one football-data.org request instead of three: the
  live window and the week of fixtures share one request, filtered by
  competition locally. Each request takes the feed several seconds.
- The serving image drops the CUDA runtime (1.77 GB → 1.09 GB).
- The README leads with verified results, screenshots and limitations; the
  milestone diary moved to `docs/history/`.

### Fixed

- A refused feed request no longer reads as "no fixtures scheduled".
- The reason a section gives for being empty is escaped before it reaches
  the page, and no longer carries Markdown that an HTML block cannot render:
  the default no-feed sentence reached the screen with its `**` still in it,
  and a live feed's reason is text the provider sent.
- The Competitions list names a shared competition name with its country
  ("Serie A — Italy", "Serie A — Brazil") instead of leaving it to the flag,
  which is what every card already did.
- The draw square in a form run reads at 4.8:1 instead of 3.9:1; WCAG AA asks
  4.5:1 of small bold text.
- A competition page and a Search result now say *why* their cards have no
  probability bars when the prediction service is down. Home said so; the
  sentence lived in the home page rather than in the call all three share.
- The package keywords dropped `mlops` and `time-series`: there is no
  retraining system, and the estimator is a tabular classifier over pre-match
  features rather than a time-series model.
- The rating-hyperparameter caveat is stated where a reviewer looks for it —
  `docs/LEAKAGE.md` now says which constants a data probe structurally cannot
  reach, and the README points at the transfer table in `docs/RATINGS.md`.
- `.gitignore` covers agent worktrees and local audit notes, which until now
  were excluded only by a `.git/info/exclude` local to one clone;
  `.dockerignore` covers them too, halving the build context.
- The dashboard no longer hangs for 15 s per call when the prediction service
  is down.
- The match page no longer crashes on matches with no recorded kick-off.
- `make dashboard` runs again, bound to 127.0.0.1.
- Webhook URLs can no longer reach the log.
- Unplayed fixtures no longer reach a Dixon-Coles refit (training and the
  backtest were unaffected).
- A local `.streamlit/secrets.toml` can no longer be copied into the dashboard
  image.
- Match cards render as one element in a browser; raw competition and
  forecaster ids no longer reach the page.
- An unreachable prediction log no longer takes the API down.
- `docs/MODELS.md` overstated how often the blend beats XGBoost per competition.
  One paragraph said 28 of 39 and another said 27; recomputed from
  `data/reports/`, it is 27. The same paragraph's median gap for the blend is
  0.0146, not the 0.0153 it carried, and the widest and narrowest competition
  gaps quoted after it are LightGBM's, which it now says.
- The 0.0163 deficit to the closing line is stated with the population it is a
  mean over. `docs/EVALUATION.md` attached it to the market table's 61,889
  matches; it is a mean over the 59,001 every forecaster could price, and over
  that table's own wider population it is 0.0168. The README's market-band
  bullet now names 61,889 rather than inheriting the 59,001 above it.

## [0.12.0] — not tagged

Milestone 12: the dashboard.

- Streamlit dashboard as an HTTP client of the API, layered
  `views → services → providers → domain`, with CI invariants for the layers.
- Match page with the forecast and how reliable forecasts of that size have
  been; a page per registered competition; the Model page.
- Every empty section names the source that would fill it.

## [0.11.0] — not tagged

Milestone 11: the inference service.

- FastAPI service over a checksum-verified artefact; `in_sample` on every
  prediction; the model card's limitations served at `/model-card/limitations`.
- Multi-stage Docker image running as a non-root user; optional PostgreSQL
  prediction log; `docker-compose.yml`.

## [0.10.0] — not tagged

Milestone 10: evaluation and explainability.

- SHAP, permutation importance and block ablation by feature block
  (`docs/EXPLAINABILITY.md`).
- Generated model card (`docs/MODEL_CARD.md`) with per-competition reliability.

## [0.9.0] — not tagged

Milestone 9: ensembling and calibration.

- Blend members admitted on error correlation; temperature scaling fitted on
  each fold's training half; reliability tables.
- Calibration halved the calibration error without moving log loss.

## [0.8.0] — not tagged

Milestone 8: the model zoo.

- Logistic regression, random forest, XGBoost, LightGBM, CatBoost and an MLP,
  tuned with Optuna on a slice before the first reported fold; MLflow tracking
  on SQLite.

## [0.7.0] — not tagged

Milestone 7: splits and baselines.

- Five expanding walk-forward folds; log loss, RPS and accuracy; four baselines
  including the de-vigged closing line.

## [0.6.0] — not tagged

Milestone 6: the leakage suite.

- Producers are discovered by walking the packages, not listed; split-boundary
  and observed-read probes; every derived column traced (`docs/LEAKAGE.md`).

## [0.5.0] — not tagged

Milestone 5: feature engineering.

- Feature registry and twenty date-cut window features; windows cut on the
  date, not the row, so same-day matches never inform each other.

## [0.4.0] — 2026-09-03

Milestone 4: ratings.

- Elo (one pool per country) and Dixon-Coles (per-competition refits), each
  with causality probes.

## [0.3.0] — 2026-09-03

Milestone 3: storage and validation.

- DuckDB views over Parquet; 24 validation checks on every ingest; the
  generated dataset card.

## [0.2.1] — 2026-09-03

- Incremental, idempotent re-runs: only new or modified provider files are
  transferred, and finished seasons can be revalidated.

## [0.2.0] — 2026-09-03

Milestone 2: ingestion.

- Provider-adapter layer for 39 competitions across 27 countries; the
  35-column canonical schema; the competition registry.

## [0.1.0] — 2026-09-03

Milestone 1: repository foundation.

- Typed configuration, logging, paths, the shared HTTP client, and the quality
  gates (ruff, black, strict mypy, pytest with coverage).
