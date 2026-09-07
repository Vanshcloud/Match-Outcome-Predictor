# Running this somewhere that is not a laptop

Milestone 16. The images and the compose file existed from Milestone 11 and 12;
what was missing was everything between "it builds" and "it is running and
somebody would know if it stopped". That is three things — a published image, a
scrape target, and a cache — and one honest list of what this still is not.

- [The images](#the-images)
- [What `docker-compose.yml` is, and what a deployment changes](#what-docker-composeyml-is-and-what-a-deployment-changes)
- [Configuration, and which of it is secret](#configuration-and-which-of-it-is-secret)
- [Health, readiness and probes](#health-readiness-and-probes)
- [Monitoring](#monitoring)
- [Caching](#caching)
- [Scaling](#scaling)
- [Updating the model](#updating-the-model)
- [What this is not](#what-this-is-not)

---

## The images

Two, from one `Dockerfile`, sharing a base, a non-root user and their mount
points. Two files describing one build is how they drift.

| | | |
|---|---|---|
| `ghcr.io/vanshcloud/match-outcome-predictor` | 1.09 GB | The API — `uvicorn`, on `$API_PORT` |
| `ghcr.io/vanshcloud/match-outcome-predictor-dashboard` | 972 MB | Streamlit, on `$DASHBOARD_PORT` |

```bash
docker pull ghcr.io/vanshcloud/match-outcome-predictor:latest
docker run -p 8000:8000 \
  -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models:ro" \
  ghcr.io/vanshcloud/match-outcome-predictor:latest
```

`.github/workflows/release.yml` publishes both on a `v*` tag. Three things
about it are deliberate:

- **It refuses a tag that disagrees with `src/__init__.py`.** A tag is a claim
  about which code an image is, and the version in that file is what `/version`
  reports and what is stamped on every row of the prediction log. Publishing
  `v0.13.0` from a tree that says `0.12.0` puts the wrong answer in an archive
  for as long as the image runs.
- **It starts both images and asserts against them before it pushes.** The same
  checks CI makes on every commit, run against the exact bytes being published
  rather than a rebuild of the same `Dockerfile`.
- **`linux/amd64` only.** The runtime image installs scipy and xgboost; a
  multi-platform build compiles what has no arm64 wheel under QEMU, which is an
  hour of emulated C++ per release for a platform nothing here deploys to.
  `docker build` on an arm64 host produces that image for anyone who needs one.

`:latest` moves only for a plain version. A pre-release — `v1.0.0-rc1` — is
published under its own name, because an unqualified `docker pull` must not
return a release candidate.

**Neither image contains data or a model.** `data/` and `models/` are mounted,
which is why retraining is `make model` and a restart rather than a rebuild,
and why the image is the same bytes across every model it serves. The dashboard
image additionally does not contain `api/` — it is a client of that service over
HTTP, and CI asserts the absence of the directory in the image as well as the
absence of the import in the source.

## What `docker-compose.yml` is, and what a deployment changes

It is a **local reproduction**, and it says so at the top of the file. It ships
a literal database password, terminates no TLS, and publishes the database port
to the host. Four things change on the way to anywhere real:

| | |
|---|---|
| `PREDICTION_LOG_DSN` | From a secret store, injected as an environment variable. It is the one secret in this project |
| TLS | Terminated in front of the API. Nothing here speaks it, on purpose — a service that terminates its own TLS is a service that has to be redeployed to rotate a certificate |
| Published ports | The API, and nothing else. Not PostgreSQL, and not the dashboard unless it is meant to be public |
| `restart` | `unless-stopped` here; whatever the orchestrator's equivalent is there |

The dashboard needs one writable path and it must not be `data/` — that is
mounted read-only on purpose, and a container that cannot corrupt its own
inputs is worth more than one that can save a preference to them. Compose gives
it a named volume at `/app/profiles`; a deployment gives it whatever survives a
restart.

## Configuration, and which of it is secret

Every variable is optional and has a working default in
`configs/config.yaml`. A real environment variable outranks that file, which is
the direction a container needs. Anything not in `ENV_OVERRIDES`
(`src/utils/config.py`) is **not** configurable by environment — an explicit map
rather than an inferred one, so a typo is not silently accepted as "not a
config variable".

| | Default | Secret |
|---|---|---|
| `API_PORT` | `8000` | |
| `API_MAX_BATCH` | `50` | |
| `API_PREDICTION_CACHE` | `1024` | |
| `DATA_DIR` / `MODEL_DIR` | `data` / `models` | |
| `LOG_LEVEL` | `INFO` | |
| `PREDICTION_LOG_DSN` | unset | **yes** — carries a password |
| `DASHBOARD_API_URL` | `http://127.0.0.1:8000` | |
| `DASHBOARD_FIXTURE_PROVIDER` | `none` | |
| `FOOTBALL_DATA_API_KEY` | unset | **yes** |
| `DASHBOARD_NOTIFIER` | `none` | |
| `DASHBOARD_WEBHOOK_URL` | unset | **yes** — anyone holding it can post as that integration |
| `DASHBOARD_PROFILE_STORE` | `<DATA_DIR>/dashboard/profiles.json` | |

The three secrets have no counterpart in `configs/config.yaml`, and that is the
mechanism rather than a convention: that file is committed, so a setting which
can only arrive by environment is one nobody can commit by accident.

**Every one of them is optional, including the secrets.** A container started
with none of them runs: the fixture feed says which variable to set, the
notifier delivers nothing, and the prediction log reports `disabled`. A
deployment that half-configures itself degrades and says which half.

## Health, readiness and probes

`/health` is **always 200 while the process is alive**, and the `status` field
says whether it can answer. That split is the whole design:

```jsonc
{
  "status": "degraded",           // "ok" | "degraded"  <- readiness
  "version": "0.12.0",
  "components": [
    {"name": "model",          "ready": false, "detail": "no model at /app/models/servable.joblib; run `make model`"},
    {"name": "fixtures",       "ready": false, "detail": "the match, ratings or feature table is missing"},
    {"name": "prediction_log", "ready": true,  "detail": "disabled"}
  ]
}
```

| Probe | Ask | Fail when |
|---|---|---|
| Liveness | `GET /health` | The request fails or times out — the process is wedged |
| Readiness | `GET /health` | `.status != "ok"` |
| Startup | `GET /health` | Same. The image allows 20 s before the first check; loading the artefact and indexing 303,517 fixtures measures 2.2 s here |

A 503 on `/health` would make "the container is wedged" and "the model has not
been built yet" indistinguishable to whatever is watching, and a person needs to
do different things about those. **The prediction log is deliberately not part
of readiness** — it is optional by design, and a readiness probe that went red
because an audit trail was switched off would pull a working forecaster out of a
load balancer for a reason unrelated to whether it can forecast.

The image declares its own `HEALTHCHECK` against the same endpoint, on
`$API_PORT`, so a bare `docker run` gets the same answer without an orchestrator.

## Monitoring

`GET /metrics`, Prometheus text exposition, no client library. The families and
what each is for:

| | | |
|---|---|---|
| `http_requests_total` | counter | `method`, `route`, `status` — the route *template*, never the URL |
| `http_request_duration_seconds_sum` / `_count` | summary | A mean per route. Not a histogram; see [API.md](API.md#get-metrics) |
| `service_ready` | gauge | `1` when a fixture can be priced. The same fact `/health`'s `status` reports |
| `service_component_ready` | gauge | Per dependency: `model`, `fixtures`, `prediction_log` |
| `fixtures_indexed` | gauge | Rows this process can price |
| `predictions_total` | counter | *Fixtures* priced, not requests — a batch of fifty is one request |
| `predictions_logged_total` | counter | Rows accepted by the prediction log |
| `prediction_cache_hits_total` | counter | Fixtures answered without calling the model |
| `prediction_cache_entries` | gauge | How full the cache is |

A scrape config that assumes nothing else:

```yaml
scrape_configs:
  - job_name: match-outcome-predictor
    metrics_path: /metrics
    static_configs:
      - targets: ["api:8000"]
```

Three alerts are worth having, and they are the three that say something a
person would act on differently:

| | Rule | Because |
|---|---|---|
| Serving without a model | `service_ready == 0` for 5 m | The service is up and answering 503. Nothing else will page |
| Silently not logging | `rate(predictions_total[15m]) > 0 and rate(predictions_logged_total[15m]) == 0` | A log failure never fails a request, on purpose. This is how it is noticed inside an hour rather than a week |
| Slow | `rate(http_request_duration_seconds_sum{route="/predict"}[5m]) / rate(http_request_duration_seconds_count{route="/predict"}[5m]) > 0.1` | 100 ms is ~13× the uncached measurement below |

Counters reset when the process does, which is the contract a scraper expects.
Nothing here is persisted.

**Logs are one line per request, as fields rather than prose**, so they grep:

```
method=POST path=/predict status=200 duration_ms=4.1 request_id=abc-123
```

An `x-request-id` header is echoed back and never invented, so a client can
quote the line it is asking about. `LOG_LEVEL=DEBUG` is the only knob; the
format is a setting nothing overrides in practice.

## Caching

Measured on the real 303,517-row table with the shipped blend:

| | Cache off | Cache on |
|---|---|---|
| `POST /predict` | 7.51 ms | **1.71 ms** |
| `POST /predict/batch`, 50 fixtures | 22.2 ms | **12.8 ms** |

Why it is safe, what is not cached, and the HTTP directives are in
[API.md](API.md#caching). The two facts a deployment needs:

- **The cache is per process.** Two replicas are two caches. There is no shared
  one on purpose — a network hop to avoid a 6 ms arithmetic operation is not a
  cache, it is a slower cache.
- **A non-2xx is never cacheable.** `/fixtures` answers 503 until the tables
  exist, and a proxy holding that for a minute would report the service down
  after it came up.

## Scaling

The service is stateless apart from three things loaded at startup — the
artefact, the fixture index and the cache — and all three are per process.

- **Horizontally**: run more containers behind one load balancer. Each unpickles
  its own artefact and indexes its own tables before it can answer — 2.2 s
  together on this machine, almost all of it the index — so a new replica needs
  its startup period honoured rather than traffic sent at it immediately.
- **Vertically**: `uvicorn --workers N` is *not* what the image does, and the
  reason is memory: each worker is another copy of the model and the 300k-row
  index. Containers scale the same way and are easier to see.
- **The prediction log is PostgreSQL rather than the DuckDB the rest of this
  project uses**, precisely because the deployment shape is several processes
  writing at once and DuckDB takes a single writer.

## Updating the model

```bash
make model                       # refit and persist the artefact (~1 min)
docker compose restart api       # or roll the deployment
```

The artefact is mounted, not baked, so this is a restart rather than a rebuild.
The restart is not optional: the model is loaded once in the lifespan and never
reloaded, which is the same property the prediction cache rests on. A service
that reloaded an artefact on demand would answer the first request after every
model change a second slower, for the rest of time.

`/version` reports what is running — the blend's members, the training window,
the number of design columns, and any library whose installed version differs
from the one that fitted the model. That last one is reported and not enforced:
a patch release of numpy will not change a forecast and refusing to start would
be the wrong call; a major one might, and never mentioning it would be the wrong
call too.

## What this is not

- **No authentication and no rate limit.** Nothing here is behind a key. The
  bound that exists is `API_MAX_BATCH`, which stops one request from asking for
  unbounded work; it is not a defence against many requests. Put this behind
  something that has one.
- **No TLS.** Terminate it in front.
- **No automatic retraining and no drift detection.** Retraining is `make model`
  and a restart, deliberately manual. Drift belongs to Milestone 19, and the
  reason is that it needs data this project does not yet have: measuring drift
  means scoring served forecasts against outcomes that arrived afterwards, which
  is exactly what the prediction log is accumulating and exactly what that
  milestone is for. A drift number computed today would be computed against the
  backtest, which is the thing drift is supposed to be measured *away from*.
- **No autoscaling policy.** `predictions_total` and the duration summary are
  the inputs one would need; what to do with them depends on a deployment this
  repository does not have.
- **Not a betting service.** The closing line beats this model in all 39
  competitions before any margin. `/model-card/limitations` says so in the
  model's own words, and every prediction carries the path to it.
