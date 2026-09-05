# The dashboard

Milestone 12. Streamlit over the measurements this project already made, with
one panel that asks the service for a live probability.

```bash
make dashboard      # http://127.0.0.1:8501
make api            # in another shell, for the "Price a fixture" tab
```

Or both, with the prediction log behind them:

```bash
docker compose up   # dashboard :8501, API :8000, PostgreSQL
```

---

## The one design decision worth reading first

**It computes nothing.**

Every table on the page is produced by `src/pipelines/report.py` or
`src/pipelines/backtest.py`, and every probability by the service over HTTP.
The dashboard chooses *which rows* and *which figure*, and nothing else.

That is not tidiness. `docs/MODEL_CARD.md` is generated from the same
functions, so a number here and the same number in the card are the same
number from the same code — and the day they disagreed, nothing would be
comparing them. A dashboard that recomputed a calibration error would be a
second measurement of the model with no test holding it to the first.

The same reasoning, one layer down, is why the predict tab uses HTTP rather
than loading `models/servable.joblib` itself. Two processes that both unpickle
the artefact are two implementations of "what does the model say". CI enforces
it: `dashboard` may not import `api`, and the dashboard image does not contain
it.

## Two data paths, and why they differ

| | Comes from | Because |
|---|---|---|
| Scoreboard, reliability, per-competition | `data/reports/ensemble/*.parquet`, read through the DuckDB store | A measurement that already exists. Reading a file is the honest way to read one, and it works with no service running. |
| Price a fixture | `POST /predict` on the running service | A live probability is the model's, and there should be exactly one process that holds the model. |

The consequence is deliberate: **the page works with the API down.** The predict
tab says the service is not answering and names the command that starts it;
every other tab is unaffected. It is the last tab for that reason.

## What each tab shows

### Scoreboard
Every forecaster over the 59,001 matches all of them could price — the common
subset, which is the only one on which two forecasters' numbers answer the same
question. `home_always` scores infinite log loss and is dropped from the bar
chart rather than clipped: a forecast that ruled out what happened was
infinitely wrong, and clipping it would state a number `src/evaluation` refuses
to.

### Reliability
**The panel the milestone exists for.** The model card reports one pooled
calibration error, 0.0015. The interesting question is what it is an average
over, and that is a filter rather than a second document — pick competitions,
pick folds, pick a bin count, and the diagram and the error move together.

The reliability diagram is the one figure in this project that a table cannot
replace. Its claim is a diagonal: a forecast stated at 0.61 should happen 61%
of the time, and `y = x` *is* the hypothesis. A reader checks the model against
it by eye in a way that a column of signed gaps does not support, because the
question is not "how large is the largest gap" but "does the curve bend, and
where".

Marker area is the matches in a bin, because the bins are wildly uneven — most
of a three-class football forecast sits between 0.15 and 0.55 — which is the
same weighting `expected_calibration_error` applies, made visible rather than
restated.

### By competition
Where the stated probabilities are least honest, worst first, beside log loss
per competition for every forecaster. Two different questions with two
different answers: the competition the model *scores* worst on is not the one
it is least *calibrated* on.

### Price a fixture
Search the fixtures the service can price, pick one, get three calibrated
probabilities. Every response carries `in_sample`, and the page shows a warning
when it is true — `make model` fits on the whole history, so today it always
is, and a reader quoting that number as the backtest's is the mistake the
boolean exists to prevent.

## Configuration

| Variable | Default | |
|---|---|---|
| `DASHBOARD_API_URL` | `http://127.0.0.1:8000` | Where the predict tab asks. Compose sets `http://api:8000` |
| `DATA_DIR` | `data` | Where the report tables are read from |
| `DASHBOARD_PORT` | `8501` | Container only; the bind address is a literal `0.0.0.0` |

There is no setting for which model the page reports on. It is
`src.models.ensemble.SHIPPED`, read from the one place that names it — the same
constant the API and the model card use.

## What it deliberately is not

- **Not a place to retrain anything.** No button here starts a fit. The
  pipelines are commands with manifests, and a fit triggered from a web page is
  a fit nobody can trace to the bytes that produced it.
- **Not a fixture list for upcoming matches.** The provider publishes results,
  not fixtures, so the matches that can be priced are the ones the batch build
  wrote. `docs/API.md` has the longer version.
- **Not authenticated, and not for public hosting.** It reads local report
  files and talks to a local service. `docker-compose.yml` is a local
  reproduction of a deployment, not a deployment — see `SECURITY.md`.
- **Not a second explainability report.** SHAP and permutation importance are
  in `docs/EXPLAINABILITY.md`; a page that recomputed them would be recomputing
  a five-minute job on a page load.

## The image

Its own stage in the same `Dockerfile`, sharing the base, the non-root user and
the mount points with the serving image — two files describing one build is how
they drift.

It installs `requirements-dashboard.txt`, which leaves out the **entire**
modelling stack: no scikit-learn, no xgboost, no joblib. This process never
unpickles an estimator, and the image is a checkable statement of that. Only
`data/` is mounted, read-only; `models/` is not mounted at all, because there
is nothing here that would open it.

CI builds it, starts it with no data, and asserts it answers `/_stcore/health`,
runs as `app`, and contains no `api` package.
