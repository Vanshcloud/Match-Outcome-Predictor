# The inference service

Milestone 11. One fixture in, three calibrated probabilities out, with the
model card's limitations reachable from the response rather than buried in a
repository.

The interactive documentation is generated from the request and response models
themselves and is at `/docs`; this page is the part a schema cannot state — why
the service can only price the fixtures it can, what a probability from it does
and does not mean, and what happens when the things it needs are not there.

---

## The one design decision worth reading first

**The service prices fixtures that are in the feature table, and it does not
compute features on demand.**

Recomputing a design row inside a request handler would be the wrong move.
Those thirty columns are produced by builders that `src/validation/leakage.py`
finds by walking the packages and probes on every build; a second
implementation living on the request path would sit outside every one of those
probes. The whole Milestone 6 argument is that a leak is invisible in the
output — it just makes the model look better — so the service reads the row the
audited pipeline wrote.

**Milestone 20 changed which fixtures are in the table, not that rule.** Until
it, the batch build wrote a row per *played* match and nothing else, so every
fixture the service could price was one the shipped artefact had trained on and
`in_sample` was `true` on every response. `make fixtures` now appends the
published fixture list to the canonical frame, runs the same ratings and
feature builders over the whole thing, and writes the served columns for the
fixtures alone — so an unplayed match has a design row built by the audited
producers, and the request path is still a dictionary lookup.

The service indexes both tables **once, in the lifespan**. A new fixture table
reaches it the way a new model does: `make fixtures`, then restart. That is
what keeps the cache directives on `/fixtures` and `/version` honest — nothing
this service says can change while the process saying it is running.

## What `in_sample` means, and why it is on every response

`make model` fits the shipped blend on the **whole** history, because that is
the model you would want to serve. Every number in
[MODEL_CARD.md](MODEL_CARD.md) is measured a different way — walk-forward, each
fold trained strictly earlier than the matches it scores.

So a probability from this service about a match the artefact was fitted on is
in-sample, and it is not the thing the 1.0156 log loss describes. One boolean
keeps those two apart. A service that omitted it would be a service whose
numbers get quoted as if they were the backtest's.

---

## Endpoints

| | | |
|---|---|---|
| `GET` | `/health` | Liveness, plus a per-component readiness breakdown |
| `GET` | `/version` | Project version, model provenance, library drift |
| `GET` | `/metrics` | Prometheus exposition: what this process has served |
| `GET` | `/model-card/limitations` | What this model must not be used for |
| `GET` | `/fixtures` | Find matches that can be priced |
| `POST` | `/predict` | One fixture, priced |
| `POST` | `/predict/batch` | Several fixtures, in one pass through the model |
| `GET` | `/docs`, `/openapi.json` | The generated documentation and schema |

### `GET /health`

**Always 200 while the process is alive.** `status` is `"ok"` or `"degraded"`.

That split is deliberate. A 503 here would make "the container is wedged" and
"the model has not been built yet" indistinguishable to whatever is watching,
and a person has to do different things about them. A load balancer reads
`status`; a human reads `components`, where each entry says what is missing and
which command produces it.

```json
{
  "status": "degraded",
  "version": "0.12.0",
  "components": [
    {"name": "model", "ready": false,
     "detail": "no model at /app/models/servable.joblib; run `make model` to fit one"},
    {"name": "fixtures", "ready": false,
     "detail": "the match, ratings or feature table is missing"},
    {"name": "prediction_log", "ready": true, "detail": "disabled"}
  ]
}
```

The prediction log is never part of `status`. It is optional by design, and a
readiness probe that went red because an audit trail was switched off would
take a working service out of rotation for a reason unrelated to whether it can
answer.

Its own `ready` flag distinguishes three states, because "nobody configured a
log" and "a log was configured and could not be reached" are different facts
and reporting both as `disabled` is how the second goes unnoticed for a week:

| `detail` | `ready` | |
|---|---|---|
| `disabled` | `true` | No `PREDICTION_LOG_DSN`. The default, and what CI runs |
| `postgres` | `true` | Connected, schema present |
| the driver's error | `false` | Configured and unreachable. `/version` says `unavailable` |

The service **starts and serves** in the third state. It has to: a node reboot
that brings the API up before PostgreSQL is exactly when a forecaster that
still answers is worth having, and a process that refused would crash-loop
until the database returned.

### `GET /version`

```json
{
  "version": "0.12.0",
  "model": "ensemble-calibrated",
  "members": ["xgboost", "logistic_regression", "mlp"],
  "design_columns": 30,
  "temperature": 1.0301,
  "trained_matches": 303517,
  "trained_from": "1993-07-23",
  "trained_through": "2026-09-01",
  "fixtures_indexed": 303517,
  "prediction_log": "postgres",
  "library_mismatches": []
}
```

Those are the values from a real fit over the full table: the temperature is
1.0301, which sits inside the 0.99–1.12 range the reported folds produced and
on the flattening side of one, as `src/models/calibration.py` says it should.

`library_mismatches` compares the versions of scikit-learn, xgboost and numpy
that *fitted* the artefact against the ones running now. It is **reported, not
enforced**: a patch release of numpy will not change a forecast and refusing to
start would be the wrong call, while a major one might and a service that never
mentioned it would also be the wrong call.

### `GET /metrics`

Milestone 16. Prometheus text exposition, `text/plain; version=0.0.4`, and no
client library — the format is a `# HELP` line, a `# TYPE` line and samples,
which is cheaper to write than `prometheus-client`'s global registry,
multiprocess mode and platform collectors are to justify on a service that
wants none of them.

```
http_requests_total{method="POST",route="/predict",status="200"} 1412
http_request_duration_seconds_sum{method="POST",route="/predict"} 3.104822
http_request_duration_seconds_count{method="POST",route="/predict"} 1412
service_ready 1
service_component_ready{component="model"} 1
service_component_ready{component="fixtures"} 1
service_component_ready{component="prediction_log"} 1
fixtures_indexed 303517
predictions_total 1461
predictions_logged_total 1461
prediction_cache_hits_total 1183
prediction_cache_entries 278
```

**The route label is the route template, never the URL.**
`/fixtures?team=Arsenal` and `/fixtures?team=Everton` are one time series. A
raw path would give every distinct query string a series of its own, which is
how a metrics endpoint becomes the largest thing a service serves; a request
that matched no route is labelled `<unmatched>`, so a scanner walking a
wordlist cannot write the wordlist into this process's memory.

**The gauges are read off the service at scrape time, not tracked.**
`service_ready` here and `status` on `/health` are two renderings of one object
rather than two records of it, so they cannot disagree.

**A sum and a count, not a histogram.** Buckets are a claim about the latency
distribution a service has. This one answers from a dictionary and a fitted
model; the honest thing to publish today is the mean, and a percentile can wait
until there is a distribution worth bucketing.

`predictions_total` counts *fixtures*, not requests: a batch of fifty is one
`http_requests_total` and fifty predictions, and the two answer different
questions.

### `GET /fixtures`

`competition_id`, `team`, `since`, `until`, `limit` (1–200). Most recent first.

Discovery exists because without it nothing else is usable — see the design
note above. Team names match past case and spacing: a provider's whitespace is
not something a caller should have to reproduce.

### `POST /predict`

Name the fixture **exactly one way**: by `match_id`, or by all four of
`competition_id`, `home_team`, `away_team` and `date`. Half of each is a 422
naming the problem, rather than a 404 implying the match is missing.

```bash
curl -sX POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"competition_id":"ENG_1","home_team":"Arsenal","away_team":"Chelsea","date":"2025-03-01"}'
```

```json
{
  "fixture": {
    "match_id": "e005dcc0757e9c04", "competition_id": "ENG_1",
    "competition": "Premier League", "country": "England", "season": "2024-25",
    "date": "2025-03-01", "home_team": "Arsenal", "away_team": "Chelsea"
  },
  "probabilities": {"home": 0.4821, "draw": 0.2604, "away": 0.2575},
  "model": "ensemble-calibrated",
  "model_version": "0.12.0",
  "in_sample": true,
  "predicted_at": "2026-09-05T09:14:02.113Z",
  "limitations_url": "/model-card/limitations"
}
```

**The fixture carries no scoreline and no odds.** The first would be answering a
different question. The second is the benchmark this project measures itself
against, and handing it back beside a forecast invites exactly the comparison
[MODEL_CARD.md](MODEL_CARD.md) says not to make.

### `POST /predict/batch`

Up to `API_MAX_BATCH` fixtures (default 50), priced in one pass through the
estimators rather than one pass each.

**Partial success.** A batch of fifty with one misspelled club returns
forty-nine forecasts and echoes the fiftieth request verbatim in `unresolved`,
so a caller can see which of theirs it was without matching on position. Only
when *nothing* resolves is it a 404.

```json
{"predictions": [ ... ], "unresolved": [{"match_id": "not a match", "...": null}], "recorded": 49}
```

`recorded` is how many rows the prediction log accepted. Zero with a log
configured is how a caller can tell it is not working; zero with no log
configured is the documented default.

### Status codes

| | |
|---|---|
| `422` | The request cannot be interpreted: a fixture named both ways or neither, an unknown field, a batch over the limit |
| `404` | The feature table holds no such fixture |
| `503` | No artefact or no feature table is loaded. The body names the missing components and the commands that produce them |
| `500` | The artefact and the feature table disagree about what a match looks like — a deployment fault, never the caller's |

Every non-2xx body is `{"detail": "..."}`, including the 422 FastAPI raises
before any handler runs. One shape, so a client writes one branch.

---

## Running it

### Locally

```bash
make reproduce   # everything, including the artefact (~60 min from empty)
make model       # just refit and persist the artefact (~1 min)
make api         # http://127.0.0.1:8000/docs
```

`make api` runs uvicorn with `--reload` and binds to loopback. The container
binds `0.0.0.0`, which is right behind a published port and wrong on a laptop.

### In Docker

```bash
make docker-build          # or: docker build -t match-outcome-predictor:local .
docker compose up          # the API and its PostgreSQL, on :8000
```

`data/` and `models/` are **mounted read-only**, never copied into the image.
Retraining is then `make model` and a restart instead of a rebuild, and the
image is the same bytes across every model it serves. The service reads a
feature table and an artefact and writes neither, so a container that cannot
corrupt its own inputs is one restart from healthy whatever else goes wrong.

The image runs as a non-root user, declares a `HEALTHCHECK` against `/health`,
and installs [`requirements-api.txt`](../requirements-api.txt) — a subset of the
runtime dependencies, with a note beside each omission saying why it is safe.
LightGBM and CatBoost are not in it: they are scored in the backtest and are
not members of the served blend. XGBoost is installed as **`xgboost-cpu`** —
the same code and the same version number without the CUDA runtime, which the
default wheel pulls in as `nvidia-*` packages measuring 291 MB inside the
image. The image is **1.09 GB**; with the default wheel it is 1.77 GB, for a
GPU nothing here has ever asked for. `xgboost-cpu` publishes the importable
`xgboost` module under a different *distribution* name, which is why
`src/pipelines/serving.py` looks for either when it records provenance.

`docker-compose.yml` is a **local reproduction of the deployment, not a
deployment**. It ships a literal password, no TLS, and a published database
port; see [SECURITY.md](../SECURITY.md). What a real one changes is in
[DEPLOYMENT.md](DEPLOYMENT.md), along with the published images and what to
scrape.

---

## Configuration

| Variable | Default | |
|---|---|---|
| `API_PORT` | `8000` | The container binds and health-checks this |
| `API_MAX_BATCH` | `50` | Fixtures per batch request |
| `API_PREDICTION_CACHE` | `1024` | Priced fixtures held before the oldest is evicted; `0` switches it off |
| `MODEL_DIR` | `models` | Where `servable.joblib` is read from |
| `DATA_DIR` | `data` | The three tables the fixture index is built from |
| `LOG_LEVEL` | `INFO` | |
| `PREDICTION_LOG_DSN` | unset | PostgreSQL for served predictions. Unset disables it |

There is no `API_HOST`. The bind address is a literal in both places that set
one — `0.0.0.0` in the image, `127.0.0.1` in `make api` — because a container
narrower than `0.0.0.0` is a mistake (the published port is how exposure is
controlled) and a configurable bind is one a health probe on loopback can be
configured out of. `API_PORT` *is* honoured, by both the server and the health
check, so `-e API_PORT=9000` moves them together.

`PREDICTION_LOG_DSN` is the one secret in this project and is **environment
only**: `configs/config.yaml` is committed, and a setting with no line in that
file is one nobody can commit by accident.

## Caching

Two kinds, and they rest on the same fact: **the artefact and the feature table
are loaded once in the lifespan and never reloaded.** A match id therefore names
one design row, which one fitted model turns into one triple of probabilities,
for the life of the process. A cache over a pure function of two immutable
things cannot serve a stale answer; it can only serve the same answer sooner.

### The prediction cache

Up to `API_PREDICTION_CACHE` priced fixtures, least-recently-used first. What it
holds back from the model is measurable on this machine, against the real
303,517-row table and the shipped blend:

| | Cache off | Cache on |
|---|---|---|
| `POST /predict`, one fixture | 7.51 ms | **1.71 ms** |
| `POST /predict/batch`, 50 fixtures | 22.2 ms | **12.8 ms** |

The residual is FastAPI's own serialisation and the index lookup, neither of
which is cached. The batch improves less because fifty fixtures were already
*one* pass through the estimators — that is what `predict/batch` is for — so
the cache removes the pass rather than forty-nine of them.

**`predicted_at` is never cached.** It is re-stamped on every response, because
it says when this service answered and not when it last did the multiplication.
The prediction log would otherwise fill with rows claiming a forecast was made
at a moment no request existed — and Milestone 19 scores that log, so an
archive whose timestamps are a cache's eviction pattern answers the wrong
question.

**The cache is per process.** Two replicas are two caches, and the hit rate
falls as replicas are added. There is deliberately no shared one: a Redis in
front of a 1.7 ms answer is a network hop to avoid an arithmetic operation.

`prediction_cache_hits_total` and `prediction_cache_entries` on `/metrics` are
how a deployment sees whether the size is right. `API_PREDICTION_CACHE=0`
switches it off, which is what a run measuring model latency wants.

### HTTP cache directives

Every response carries `Cache-Control`. Three routes are reusable:

| Route | Directive | Because |
|---|---|---|
| `/version` | `public, max-age=300` | The process cannot change what it was fitted on |
| `/model-card/limitations` | `public, max-age=3600` | The card's own constant |
| `/fixtures` | `public, max-age=60` | The index is built once at startup |
| everything else | `no-store` | |

`no-store` is the important half. A cached `/health` is a proxy answering a
liveness question on behalf of a process it has not spoken to, and a cached
`/metrics` is a counter that appears to stop — both are failures that *look
like* health. And a non-2xx is never cacheable whatever its route: `/fixtures`
answers 503 until the tables are built, which is the one thing about it that is
not stable for the life of the process.

There is no `ETag`. A conditional request still costs the round trip, and these
bodies are hundreds of bytes; `max-age` removes the round trip, which is the
part worth having.

## Logging

One line per request, as fields rather than prose, so it greps:

```
method=POST path=/predict status=200 duration_ms=4.1 request_id=abc-123
```

An `x-request-id` header is echoed back on the response, so a client can quote
the line it is asking about. One is never invented — a request without an id
gets no header.

## The prediction log

Every served prediction is written with the inputs it was made from: the match,
the model, the three probabilities, `in_sample`, and when. The outcome is
deliberately **not** a column — it is in the canonical match table already, and
a consumer joins on `match_id`.

The reason it exists is that the backtest measures the model over folds and a
*served* model is a different thing: a different training window, a different
population of requests, and outcomes that arrive later. Writing predictions down
is what will let the served calibration be measured against what happened rather
than assumed to match the backtest.

**A log failure never fails a request, and never fails a startup.** A
prediction answered and not logged is a gap in an audit trail; a prediction not
answered because the audit trail was down is an outage. The first is the lesser
harm, so a rejected write is logged at error level and the response goes out
with `recorded: 0` — and a database that cannot be reached when the process
starts is caught the same way the missing model and the missing tables are,
rather than raised out of the lifespan.

PostgreSQL rather than the DuckDB the rest of the project uses, because DuckDB
takes a single writer and the deployment shape is several uvicorn workers behind
one port.

```bash
docker compose up -d postgres
PREDICTION_LOG_DSN=postgresql://predictor:predictor@127.0.0.1:5432/predictions \
  make test-int          # runs the log's integration tests against it
```

---

## What the service is not

- **Not a betting service.** The closing line beats this model in all 39
  competitions before any margin is taken off. `/model-card/limitations` says so
  in the model's own words, and every prediction carries the path to it.
- **Not live or in-play.** Every input is knowable before kick-off by
  construction, and nothing updates during a match.
- **Not an explanation.** These probabilities are ranked, not reasoned;
  [EXPLAINABILITY.md](EXPLAINABILITY.md) says which blocks the model leans on
  across many matches, which is not a claim about any one fixture.

## Where the code is

```
api/
  main.py       the application: lifespan, middleware, error mapping, cache policy
  routes.py     seven endpoints, each a call and a return
  service.py    what answers a request: model, index, log, prediction cache
  metrics.py    the counters, and the exposition rendered from them
  schemas.py    the request and response models, which are also the OpenAPI doc
src/models/artifact.py     the shipped blend, fitted once — frames in, arrays out
src/pipelines/serving.py   persisting it, loading it, and finding a fixture
src/storage/predictions.py the prediction log: PostgreSQL, and the no-op
scripts/build_model.py     `make model`
```

CI enforces the seam in both directions: nothing in `src` imports `api`, and
`api` may reach the pipelines, models, storage, evaluation and utils — and not
`src.ratings`, `src.feature_engineering` or `src.ingestion`, which is the
shortcut that would put an unprobed copy of the feature layer on the request
path.
