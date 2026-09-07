# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

One entry per milestone. An entry says what changed and, where the choice was
not obvious, why — a changelog that only lists filenames is a `git log` with
extra steps.

## [Unreleased]

### Added

- **Milestone 17 — the closing line, and what a gap from it measures.** The
  odds have been in the canonical table since Milestone 2 and off every page
  since Milestone 12, on the stated grounds that *"showing both invites the
  comparison to be made without the walk-forward folds that make it
  meaningful."* This milestone satisfies that objection rather than overruling
  it: the comparison is made **with** the folds, and the page shows the answer.

  The answer is not the one the layout implies. Over 61,889 out-of-sample
  forecasts, grouped by how far the model was from the price:

  | Apart | n | Model | Market | Model − market | Model better |
  |---|---:|---:|---:|---:|---:|
  | <2% | 9,628 | 0.9881 | 0.9872 | **+0.0009** | 49.2% |
  | 2–5% | 21,139 | 1.0136 | 1.0099 | +0.0037 | 48.4% |
  | 5–10% | 20,874 | 1.0199 | 1.0059 | +0.0140 | 46.4% |
  | 10–20% | 9,419 | 1.0402 | 0.9864 | +0.0538 | 41.9% |
  | >20% | 829 | 1.0810 | 0.8975 | **+0.1835** | 35.5% |

  **Where the model agrees with the closing line it is level with it** —
  +0.0009 over 9,628 matches — so the 0.0163 project-level deficit reported
  since Milestone 9 is not spread thinly across every match. It is concentrated
  in the ones the model sees differently, and it grows by a factor of about 200
  from the narrowest band to the widest. In that widest band the market's own
  log loss *falls* to 0.8975: those 829 matches are ones it prices confidently
  and correctly while the model does not.

  That rules out the reading a value detector rests on, so **no value detector
  was built.** A gap is presented as what it measures — this model's likely
  error on that fixture — and no page suggests a bet, flags value or computes
  an expected return. The restraint is a measurement rather than caution.

  **`src/evaluation/market.py`** is new and holds both ends of the subject: the
  de-vig, moved out of the `bookmaker` baseline because three callers now need
  "what did the market say" and three implementations of removing an overround
  is three chances to publish a number that is not the one the benchmark was
  scored against; and `disagreement`, which produces the table above.
  `src/pipelines/report.py::market_comparison` joins it to the forecasts and
  `make card` writes `market.parquet` beside them.

  **`fold_forecasts` now carries `match_id`.** One line, and it is what makes
  the join a join. The alternative — re-deriving the fold split wherever the
  odds are needed and trusting it to produce the same row order — is a
  plausible-looking wrong answer waiting for a split parameter to change. The
  model card is byte-identical after the regeneration: no model, metric or
  reported number moved.

  **The dashboard gained a fourth provider protocol**, which is exactly what
  Milestone 12's estimate said this would cost. `OddsProvider`, satisfied by
  `HistoricalOdds` over the same canonical table the results come from, one
  field on the context, and no view moved. Two panels replace two placeholders
  on the match page: what the market said (with the overround stated, about 8%
  in this feed) and how far apart the two are, with the band's measured verdict
  under it.

  **Expected goals ship as the goal model's own rates**, and the label is the
  point. `dc_home_lambda` and `dc_away_lambda` are the two Poisson rates
  Milestone 4's Dixon-Coles model fits, and they were already in the ratings
  table — the placeholder that used to sit in that panel claimed this project
  "fits none", which was simply wrong. They are not xG off a shot map, nothing
  here ingests one, and the panel says so rather than blurring it.

  One defect found on the way: reading three nullable `Float64` cells out of a
  *row* of the match table yields an object array, and a missing price in one
  is a `NAType` that `float()` refuses. About a fifth of the table has no
  price, so that was the ordinary path rather than an edge case; the read is
  off the one-row frame now.

- **Milestone 16 — deployment and operations.** Three things stood between "it
  builds" and "it is running and somebody would know if it stopped": a
  published image, a scrape target, and a cache.

  **`GET /metrics`**, Prometheus text exposition, and no client library.
  `prometheus-client` renders this format and also brings a process-global
  registry, a multiprocess mode, a WSGI app and platform collectors — none of
  which this service wants — for a format that is a `# HELP` line, a `# TYPE`
  line and samples. `api/metrics.py` is 46 statements and adds no third-party
  import to the request path.

  The label is the **route template**, off `request.scope`, never the URL.
  `/fixtures?team=Arsenal` and `/fixtures?team=Everton` are one time series;
  a request that matched no route is one series called `<unmatched>`, so a
  scanner walking a wordlist cannot write the wordlist into this process's
  memory. The gauges — `service_ready`, `service_component_ready` — are read
  off the service at scrape time rather than tracked, so they cannot disagree
  with what `/health` says: two renderings of one object rather than two
  records of it. Latency is a sum and a count, not a histogram, because buckets
  are a claim about a distribution and the honest thing to publish today is the
  mean. `predictions_total` counts *fixtures*, not requests — a batch of fifty
  is one request and fifty forecasts, and the two answer different questions.

  **A prediction cache**, `API_PREDICTION_CACHE` entries, least-recently-used.
  It rests on a property rather than a hope: the artefact and the feature table
  are loaded once in the lifespan and never reloaded, so a match id names one
  design row that one fitted model turns into one triple of probabilities for
  the life of the process. A cache over a pure function of two immutable things
  cannot serve a stale answer. Measured against the real 303,517-row table:
  `POST /predict` 7.51 ms → **1.71 ms**, a batch of fifty 22.2 ms → **12.8 ms**.

  **`predicted_at` is deliberately not cached** and is re-stamped on every
  response, because it says when this service answered and not when it last did
  the multiplication. The prediction log would otherwise fill with rows claiming
  a forecast was made at a moment no request existed — and Milestone 19 scores
  that log, so an archive whose timestamps are a cache's eviction pattern
  answers the wrong question. The answers are also read back out of a local
  mapping rather than back through the cache, because an entry stored while
  pricing a batch can be evicted by a later entry in the *same* batch.

  **Every response carries `Cache-Control`.** Three routes are reusable for the
  reason above; everything else is `no-store`, which is the half that matters —
  a cached `/health` is a proxy answering a liveness question on behalf of a
  process it has not spoken to, and a cached `/metrics` is a counter that
  appears to stop. Both are failures that look like health. A non-2xx is never
  cacheable whatever its route: `/fixtures` answers 503 until the tables exist,
  and a proxy holding that for a minute would report the service down after it
  came up. No `ETag` — a conditional request still costs the round trip, and
  `max-age` removes it.

  **`.github/workflows/release.yml`** publishes both images to GHCR on a `v*`
  tag, with no third-party actions: an action that moves under a release
  workflow moves the bytes of a published artefact with it. It **refuses a tag
  that disagrees with `src/__init__.py`** — that version is what `/version`
  reports and what is stamped on every row of the prediction log, so a
  mislabelled image puts the wrong answer in an archive for as long as it runs
  — and it starts both images and asserts against them *before* pushing, so
  what is published is what was tested rather than a rebuild of the same
  Dockerfile. `linux/amd64` only: a multi-platform build compiles scipy and
  xgboost under QEMU for a platform nothing here deploys to. `:latest` moves
  only for a plain version, never a pre-release.

  **Drift and retraining were deferred to Milestone 19 rather than built.** The
  README listed them here. Measuring drift means scoring served forecasts
  against outcomes that arrived afterwards, which is precisely what the
  prediction log is accumulating and precisely what Milestone 19 is for; a
  drift number computed today would be computed against the backtest, which is
  the thing drift is supposed to be measured away from.

  `docs/DEPLOYMENT.md` is new: the published images, what compose is not, the
  probe table, a scrape config, three alerts worth having, and the list of what
  this still is not — no authentication, no rate limit, no TLS, no autoscaling
  policy.

- **Milestone 15 — live tracking and alerts.** The live strip repaints itself
  every sixty seconds (`@st.fragment(run_every=…)`) and says what changed since
  it last looked: a kick-off, a goal, a final whistle. Each is a toast in the
  page and, where a transport is configured, a POST to it.

  Milestone 12 estimated this as "a second method on the fixture feed". The
  card estimate held — no card changed — but the method did not: the feed
  already had `live()`, and what was missing was *memory*.
  `dashboard/services/watch.py` keeps one snapshot per browser tab and diffs
  two looks. Per tab, because "since *I* last looked" is a per-tab question and
  a snapshot in the profile store would mean the first tab to refresh silently
  consumed the second one's news.

  Two rules in that diff are wrong in the obvious implementation. **The first
  look announces nothing** — with no previous snapshot there is no "since", and
  a toast reading "kick-off" for a match already an hour old tells a reader
  something untrue. **An empty answer is not full time** — the feed returns
  nothing both when nothing is in play and when it could not be reached, so
  absence read as "the match ended" would announce eight final whistles because
  of one rate limit; a look the feed did not answer produces no events at all.
  A changing *minute* is deliberately not an event, which is also why the free
  tier's missing `minute` field costs this milestone nothing.

  Full time turned out to be a match that has *gone* from the answer rather
  than one whose status changed, because `live()` returns what is in play — so
  a snapshot holds fixtures rather than a status string: by the time a match is
  over, the snapshot is the only record of the score it finished on.

  The transport is `providers/webhook.py` behind a new `Notifier` protocol,
  with `NullNotifier` as the default — the same registry-and-environment-variable
  shape as the fixture feed. One body serves the common receivers (`text` for
  Slack, `content` for Discord, the event's own fields for anything
  programmatic), so there is no "which flavour of webhook" setting. A transport
  that refuses is a `False` and a caption, never an exception, and nothing is
  retried: a POST that failed may already have been acted on, so a duplicate
  goal alert cannot be this dashboard's doing.

  **Alerts exist while something is watching** — the page has to be open. A
  process that polls with every browser closed is a different thing with its
  own lifecycle, and it was left out rather than half-built.

  Verified against two matches genuinely in play, with a local webhook
  receiver: the first look announced nothing, a rewound snapshot produced
  exactly one `goal` event, and a match removed from the answer produced one
  `full-time` carrying the score it finished on. Two dead branches were deleted
  rather than tested — a branch no test can honestly reach is a branch that
  should not exist.

  CI's session-state rule was restated in the process. Milestone 14 wrote it as
  "exactly one module", which was stricter than the property it protects; it now
  checks that property directly — `dashboard/views/` never touches session
  state, and outside the views only the two modules whose whole job is
  per-reader state may.

- **Milestone 14 — accounts and saved favourites.** Favourites used to live in
  `st.session_state`: per browser tab, gone when it closed. They now live in a
  JSON file keyed by whoever the reader is, and Milestone 12's estimate held —
  four accessors in `dashboard/domain/favourites.py` changed, every signature
  stayed, and no view was touched.

  `dashboard/domain/identity.py` is the only module that knows how a reader is
  named, and there are two answers: a **profile** (`profile:<name>`, picked
  from the sidebar, `Guest` by default and a real row rather than a null case,
  so favourites persist for someone who never opens the picker) and an
  **account** (`account:<verified email>`, where a deployment configures OIDC
  for Streamlit's `st.login()`). The keys are namespaced because without the
  prefixes a profile named after a colleague's email address would be handed
  that colleague's favourites.

  Accounts are off unless configured, and the sidebar offers the button only
  when signing in would actually work. Three things about Streamlit's auth
  surface were checked rather than assumed: `st.user.is_logged_in` *raises*
  with no provider configured (the key exists only when `secrets.toml` has an
  `[auth]` section, which is how "is a provider configured" is answered without
  touching `st.secrets`, which raises when there is no file); and `st.login()`
  raises without `streamlit[auth]`, which the dashboard image deliberately does
  not install.

  The store is a file rather than the PostgreSQL already in the compose file —
  that database is the service's, and reaching it from here would mean `psycopg`
  in an image documenting its absence, a pool nobody tracks across reruns, and
  a migration for a preference. Written through a temporary file and an atomic
  rename, because a partial JSON document is unreadable and would be read on
  the next page load. A write that cannot land is a sentence in the sidebar
  rather than an exception: `data/` is mounted read-only in compose, so it is a
  state a real deployment reaches. Its path is `$DASHBOARD_PROFILE_STORE`, and
  compose points it at a writable named volume.

  Two things this milestone hardened rather than added. CI now asserts that
  `session_state` is used in exactly one module — Milestone 12 claimed that
  property and nothing checked it, and it is the reason accounts were three new
  files instead of a search through six pages; the rule was confirmed to bite
  by planting a violation in a view. And `tests/conftest.py` now points the
  store at a temporary file for every test, autouse, after the first suite run
  wrote real profiles into the developer's `data/` directory and later tests
  began inheriting clubs that earlier ones had followed.

- **Milestone 13 — live fixtures.** `dashboard/providers/football_data_org.py`
  reads `GET /v4/matches` from football-data.org and answers the three
  questions this project's results feed cannot: what is on today, what is on
  this week, and what is being played right now. Selecting it is
  `DASHBOARD_FIXTURE_PROVIDER=football-data.org` and `FOOTBALL_DATA_API_KEY`;
  with neither, the null feed still answers and the sections still say why they
  are empty.

  The milestone cost what Milestone 12 said it would — one class, one registry
  entry, one environment variable — and no view moved. The single view line
  that changed is the sidebar caption, which used to read `✕ No fixture feed —
  Milestone 13` and now names the feed that answered or the provider's own
  reason for having nothing; a status bar citing an unshipped milestone after
  it ships is a small lie a reader stops checking the rest of the page against.

  What it deliberately is **not**:

  - **Not a second ingestion source.** Nothing it returns is written to a
    table, joined to one, or read by a model. No model, feature, split, metric
    or reported number changed, and `matches.parquet` holds exactly the rows it
    held before.
  - **Not joinable, and its ids say so.** `fdorg-497821` rather than a
    `make_match_id` hash, because that id includes the team names as its source
    spells them and this feed says "Manchester United FC" where the ingested
    table says "Man United". An id that looked canonical and matched nothing
    would be worse than one that names its origin. `src/ingestion/teams.py`'s
    alias table stays empty for the same reason: it has no reader until a
    milestone actually *ingests* a second vocabulary.
  - **Not all thirty-nine competitions.** Nine, which is the free tier. A
    followed competition outside them is dropped from the filter, because this
    project's ids are its own and there is no code to send for it, and a reader
    following only such competitions is told so.
  - **Not a source of fixtures it does not have.** Postponed, suspended and
    cancelled matches are dropped rather than rendered as scheduled, a row with
    no parseable kick-off or no named teams is dropped, and a crest URL that is
    not `https` is not put in an `img` tag.

  Three defects the live API found that review had not, all fixed before the
  milestone was called done:

  - **The feed's calendar is UTC and the dashboard's is the host's.** At
    UTC+05:30 those are different days for five and a half hours out of every
    twenty-four, so `live()` asked for the wrong date and would have shown
    nothing through Saturday evening in Europe. Fixtures now carry host-local
    date and kick-off, the live window is anchored to UTC and filtered by
    status rather than by date, and a requested window is widened a day at each
    end and narrowed back in the answer. The regression test pins `TZ`, because
    in UTC — which is what CI runs in — the bug does not reproduce.
  - **A window wider than ten days is a `400`.** A fourteen-day ask came back
    empty. Longer windows are split into consecutive requests rather than
    truncated.
  - **`dateTo` is an instant, not a day.** The feed's window is
    `[dateFrom T00:00Z, dateTo T00:00Z]`, so a same-day window answers nothing
    and "yesterday to today" excludes today. `live()` returned nothing while
    two matches were in play, and would have done so permanently. Dates in the
    module are inclusive days now, converted in one place at the wire — and
    because consecutive chunks then share an instant, a midnight kick-off (every
    Brazilian evening match) is merged by `match_id` instead of appearing
    twice.
  - **This feed's errors are in the body, not the status line.** It sends an
    empty reason phrase, so an invalid token and an over-wide window both read
    `answered 400: `. The JSON `message` is now what a reader is shown.

  Two smaller decisions worth recording. `available` makes the same
  today-window request `live()` does, so a refused key renders as a key problem
  instead of as "no fixtures scheduled in the next week" — which is a statement
  about football. And answers are memoised for sixty seconds by an `lru_cache`
  keyed on a time bucket at module level: the free tier allows ten calls a
  minute, Streamlit reruns the script on every click, and the context builds a
  fresh provider on every rerun, so an instance cache would be empty every time
  it was read.

The rest of this entry is the full independent audit of the repository that
preceded the milestone above. No modelling code, feature, split, metric or
reported number changed in either; every fix is on the serving and packaging
side, and the walk-forward tables are the same tables.

### Fixed

- **The dashboard no longer reports a healthy prediction service when an
  unrelated one is on the port.** `DASHBOARD_API_URL` defaults to
  `http://127.0.0.1:8000`, a different project's API was listening there, and
  its `/health` answered `{"status": "ok"}` — which is all `ApiPredictions`
  checked. The sidebar therefore showed `✓ Prediction service` while every
  `/predict` would have come back 404, and the failure would have surfaced as
  an unexplained error on the match page rather than as the misconfiguration it
  was. `available` now also requires the body to be the document
  `api/schemas.py` describes: `components` is a required field of
  `HealthResponse` and is present whether the service is ready or degraded, so
  its absence means something else is on the port. The caption says so and
  names the variable to set.

  Checked by shape rather than by matching component names, because the set of
  components is a thing later milestones add to and a check that enumerated
  them would fail on the milestone that adds one. Found by running the
  dashboard, not by a test — a liveness probe that accepts any 200 is a probe
  for "something is listening", which is not the question anyone was asking.

- **An unreachable prediction log no longer takes the service down.**
  `open_prediction_log` creates its table on the way up, so a configured
  PostgreSQL that was down raised out of the lifespan and the process
  crash-looped — while every docstring in the serving layer said the log was
  optional and that a log which refuses never fails a request. The model and
  the fixture index were each already guarded; the log was not. It is now
  caught like the other two, `/health` reports the component as not ready with
  the driver's own reason, and `/version` says `unavailable` rather than
  `disabled`, because a log nobody configured and a log that could not be
  reached are different facts. The regression test asserts the process serves
  predictions with the database refused.
- **`/version` no longer answers 500 when a library is installed under another
  distribution name.** The provenance lookup asked `importlib.metadata` for
  `xgboost` and let `PackageNotFoundError` escape. A library nothing provides
  is now recorded as `"unknown"`, which the drift check reads as a mismatch —
  the warning that was wanted — instead of failing the one endpoint whose job
  is to say what is running.
- **`uv.lock` could not reconstruct a runnable service.** It was generated
  before the Milestone 11 commit that added the API dependencies to
  `requirements.txt`, so the project's own dependency list omitted `fastapi`,
  `joblib` and `uvicorn`, and `psycopg` was absent from the file entirely — no
  package entry, no hashes. `uv sync --frozen` would therefore have produced an
  environment with no driver for the prediction log, against a README that
  offers the lockfile as the way to reconstruct the environment the benchmark
  numbers were measured in. The rule to regenerate it in the same commit as any
  requirements change was already written down; this is the commit it was
  missed in, applied late.

### Changed

- **The serving image drops the CUDA runtime: 1.77 GB to 1.09 GB.**
  `requirements-api.txt` installs `xgboost-cpu` rather than `xgboost`. The
  default wheel depends on `nvidia-*` packages that measure 291 MB inside the
  image, for a service that scores three classes on a CPU and has never asked
  for a GPU. Verified by serving the same artefact from both images and
  comparing: bit-identical probabilities over 25 fixtures, and
  `library_mismatches` still empty, because the CPU build reports the same
  version.
- **The fixture index holds the columns the service reads, and no others.**
  The join is shared with the training pipeline, which needs the whole
  canonical table; serving needs the eight columns a fixture is described by
  and the thirty the model prices. The twenty-seven dropped include the
  scoreline and the three odds columns, so the process answering requests does
  not hold the benchmark the model is measured against. Not a memory
  optimisation, and the docstring says so: the frame falls from 329 MB to
  185 MB and process RSS does not move.
- **`API_HOST` is gone.** It was mapped in `ENV_OVERRIDES`, listed in
  `.env.example`, set in `configs/config.yaml` and read by nothing — the
  Makefile binds `127.0.0.1` and the image binds `0.0.0.0`, both as literals.
  A setting that appears to be configured and is not is the exact failure
  `extra="forbid"` exists to prevent. `API_PORT` is unaffected.
- **`urllib3` is declared.** `src/utils/http.py` imports `urllib3.util.Retry`
  directly and the dependency arrived only via `requests` — an import
  satisfied by somebody else's dependency is one that breaks on the release
  where they drop it.
- **The OpenAPI document describes its error bodies.** `ErrorResponse` was
  defined in `api/schemas.py` and referenced nowhere, so every non-2xx
  response was documented with a description and no schema. It is now attached
  to all seven declared error statuses.
- `docs/EVALUATION.md` records a measurement the repository had not made: the
  gap to the closing line is 0.0134 in the ten competitions carrying shot data
  and 0.0177 in the twenty-nine carrying none — 32% wider where the provider
  publishes less, at r = -0.36 and p = 0.045. That is the first direct evidence
  for the claim this project makes three times, and the section is explicit
  that a marginal p over ten competitions against twenty-nine, confounded with
  league maturity, points the way the argument does without settling it.

## [0.12.0] — unreleased

Milestone 12: the dashboard. A football dashboard over a prediction engine it
is only ever a client of — six pages, four layers, and every empty section
naming the provider that would fill it.

### Added

- `dashboard/` — a new top-level package and the top of the dependency graph,
  arranged in **four layers that point one way**:
  `views → services → providers → domain`.
  - `domain/` — what a match, a forecast and a competition *are*. Value types
    with no I/O, so a provider, a test and one day a different front end all
    mean the same thing by a `Fixture`.
  - `providers/` — where football comes from. One protocol per kind of source
    (`ResultProvider`, `FixtureProvider`, `PredictionProvider`) and one
    implementation per source: the canonical table for results, the service
    over HTTP for forecasts, and a null feed for fixtures.
  - `services/` — orchestration and caching. What a page *needs*, assembled
    from three providers and a favourites list.
  - `views/` — Streamlit, thin, and the only layer that would be rewritten if
    this became a React client reading the same API.
- **Six pages**, declared through `st.navigation` rather than discovered from a
  `pages/` directory: Home, Live centre, Competitions, Match, Search and Model.
  Declared, because a page then gets a stable `url_path` a card can deep-link
  to with a query parameter, and every view stays an ordinary function a test
  can run instead of a script only Streamlit knows how to execute.
- **The match page reports what a probability is worth.** Beside three
  calibrated probabilities it shows how often forecasts stated in the same band
  actually happened, in that competition, from the same reliability tables
  `docs/MODEL_CARD.md` is generated from. A stated probability with no measured
  reliability beside it is the number this project exists to stop people
  quoting. Where no band was measured — a competition the backtest never
  covered, or a probability outside the measured range — the page says so
  rather than reaching for the nearest bin.
- **A competition page per registered league, and nobody wrote them.** The page
  is a function of `configs/leagues.yaml`, which is the same property
  `tests/unit/test_registry.py` asserts about ingestion, extended to the
  presentation layer.
- **Favourites**, in `st.session_state`. Every page reads them; none of them
  touches `st.session_state`, which is the whole of the Milestone 14 seam.
  Following no league means *all* the football, not none of it, and that
  distinction is made in exactly one function.
- `src/pipelines/tables.py::read_matches` and `RESULT_COLUMNS` — finished
  matches, projected to what a reader is shown. The only addition to `src` this
  milestone makes, and it reads: no model, feature, split, metric or reported
  number changed. The projection deliberately excludes the three odds columns
  and the thirty design columns.
- **It computes nothing, and that is the design.** Every table comes from
  `src/pipelines/report.py` or `src/pipelines/backtest.py` and every
  probability from the service. `docs/MODEL_CARD.md` is generated from the same
  functions, so a number on the page and the same number in the card are the
  same number from the same code — a dashboard with its own copy would be a
  second measurement of the model with nothing holding it to the first.
- **Two data paths, deliberately.** Reports are read from disk; predictions
  come over HTTP. Two processes that both unpickle the artefact are two
  implementations of "what does the model say", and the day they disagreed
  nothing would be comparing them. The consequence is that the page works with
  the API down — the predict tab says so and names the command that starts it,
  and the other three tabs are unaffected. It is the last tab for that reason.
- `src/pipelines/report.py::write_forecasts` and
  `src/pipelines/tables.py::read_forecasts` — the per-match diagnostic pass,
  persisted. `make card` computed 62,036 forecasts, used them and threw them
  away; the dashboard needs the same rows on every page load and would
  otherwise have spent five minutes on each. Written through the same
  `persist` every derived table uses, so the provenance of a reliability
  diagram is checkable the way the provenance of a rating is.
- `HttpClient.post`. The comment on `RETRY_METHODS` anticipated it: POST
  inherits the restriction rather than widening it, so a failed prediction
  comes back as one error rather than four more attempts at a request the
  server may already have acted on.
- `DashboardConfig` — `api_url` and `request_timeout_seconds`. The URL is a
  setting rather than a literal because compose overrides it to
  `http://api:8000`, which is the whole reason it exists.
- A dashboard stage in the `Dockerfile` and a service in `docker-compose.yml`.
  `docker compose up` now brings up the API, PostgreSQL and the dashboard.
- `docs/DASHBOARD.md`, `make dashboard`, `requirements-dashboard.txt`, and
  `.streamlit/config.toml` — Streamlit's own dark theme for the chrome, with
  `dashboard/theme.py` adding only what it has no setting for: the cards.
- **A component layer**, `dashboard/ui.py`: match cards, probability bars,
  generated crests, form strings. Cards are anchors rather than buttons,
  because a grid of forty `st.button` widgets is forty round trips to the
  server. Every name that came out of a provider's CSV is escaped once, here.
  Crests are the club's initials on a colour derived from its name — stable
  between sessions, because a reader scanning forty cards navigates by colour
  before they read a word.
- **Six CI invariants.** The four from the first pass — `dashboard` may not
  import `api`; nothing may import `dashboard`; the dashboard may not reach
  past the reporting layer; the image job builds it, starts it with no data and
  asserts it is healthy, runs as `app` and contains no `api` package — plus two
  for the new shape: the layers point one way, and **no view names a
  provider**. That last one is the property the provider layer exists for; the
  day a view imports a concrete provider, connecting a real feed stops being
  one class and becomes a search.
- `DASHBOARD_FIXTURE_PROVIDER`, in `.env.example` and `docker-compose.yml`.

### Changed

- Coverage, lint, format and type-checking cover `dashboard` as well as `src`
  and `api`. The threshold is still 100%.
- The HTTP-confinement invariant now forbids **call sites** rather than the
  `requests` import. `dashboard/client.py` imports `requests` for its exception
  tree — it catches `RequestException` and types a `Response` — while making
  every request through `HttpClient`, and a rule banning the import would have
  been asking it to catch exceptions it cannot name. The transports with no
  exception-only use are still banned outright. Verified against a planted
  `requests.get`.
- `plotly-stubs` pinned in `requirements-lint.txt`. It earned its place
  immediately by catching a `dict[str, object]` splatted into `update_layout`,
  which type-checks as nothing in particular and is how a misspelt layout key
  survives to be ignored at run time. streamlit ships its own `py.typed` and
  needs neither a stub nor an override.

### Fixed

- **`streamlit run dashboard/app.py` died on the first browser session.**
  Streamlit puts the *script's own directory* on `sys.path`, not the project
  root, so `import dashboard` had nothing to resolve against and every local
  run failed with `ModuleNotFoundError: No module named 'dashboard'` the moment
  a browser connected. `src` is installed as a package and `dashboard` is not,
  which is why one resolved and the other did not.

  It was invisible to everything that was watching. The container sets
  `PYTHONPATH=/app`, so the image was green; pytest puts the rootdir on the
  path before any test runs, so `AppTest` was green; and the CI image check
  asks `/_stcore/health`, which answers before the script executes. Three
  green signals and a command that had never worked.

  The root is now prepended by `dashboard/app.py` itself rather than by the
  `make` target, because `streamlit run dashboard/app.py` is the documented
  command and people type it directly — a target that exported `PYTHONPATH`
  would have fixed the invocation that already had a wrapper and left the bare
  one broken. The regression test runs the entry point under an isolated
  interpreter with only `dashboard/` on the path, which is the environment the
  server actually gives it, and fails without the fix.

### The three things it cannot show, and why they are still on the page

Today's matches, upcoming fixtures and live scores need a fixture feed, and
this project ingests **results** — a match that has not been played is in no
table here. Those sections are rendered from `NullFixtures`, a real
implementation of `FixtureProvider` that returns nothing and carries its own
reason, which the page prints.

Nothing invents a fixture. Plausible-looking generated matches would put a game
on the screen that is not being played, and that is the one failure this
application cannot recover from: a reader who catches it once stops believing
the real rows too.

Connecting a feed is a class satisfying the protocol, an entry in
`FIXTURE_PROVIDERS`, and one environment variable. No view moves.
`tests/unit/test_dashboard_app.py` rehearses it with a stub feed, so the claim
is a measurement rather than an aspiration.

### Deviations from the plan, with reasons

- **The plan left the data path open — "a client of `api/` or of
  `src/pipelines`" — and the answer is both.** Neither alone works. The API
  exposes no reliability or backtest data, so a pure HTTP client would have
  needed three new endpoints this milestone did not ask for; and a pure `src`
  client would have had to load the artefact, which is the second copy of the
  model the whole arrangement exists to avoid.
- **One figure, not a chart library.** `requirements.txt` recorded at Milestone
  10 that matplotlib was left out because every figure it would have drawn was
  a five-row table. That still holds for four of the five things on this page.
  The reliability diagram is the exception, and the reason is specific: its
  claim is a diagonal, `y = x` *is* the hypothesis, and a reader checks a
  forecast against it by eye in a way a column of signed gaps does not support.
- **Streamlit rather than a React client**, and the layering is what makes that
  reversible rather than a bet. A React front end reading the same API would
  replace `views/` and keep `domain/`, `providers/` and `services/` — which is
  why the caching lives in the services and the providers hold no state.
- **Standings are absent deliberately.** A league table is a season's results
  added up and this project has the results, but a table that is *right* needs
  each competition's own rules for points, tie-breaks, deductions and
  play-offs, and thirty-nine competitions do not share them. A table that
  silently ranked Argentina by goal difference when Argentina does not would be
  worse than no table.
- **Odds stay off every page.** They are in the match table and they are the
  benchmark this project measures itself against; showing them beside a
  forecast invites the comparison to be made without the walk-forward folds
  that make it meaningful. `docs/EVALUATION.md` is where that comparison lives.

## [0.11.0] — unreleased

Milestone 11: the inference service. The shipped blend, fitted once and served
over HTTP from a container, with the model card's limitations reachable from
the response and every prediction saying whether it was in-sample.

### Added

- `api/` — a new top-level package, and a caller of `src` rather than a part of
  it. Six endpoints: `/predict` and `/predict/batch`, `/fixtures`, `/health`,
  `/version`, `/model-card/limitations`. The request and response models *are*
  the OpenAPI document, so a field that changes shape cannot leave the
  documentation describing the old one.
- `src/models/artifact.py` — the fitted counterpart of the blend. Every
  forecaster before this one fits inside its own `forecast(train, evaluate)`,
  which is what makes the folds causal and exactly the wrong shape for a
  process that answers requests. Same members, same mean, same temperature
  fitted the same way — called rather than reimplemented, because a served
  probability that differed from a backtested one would make every number in
  `docs/MODEL_CARD.md` a claim about a different model. Frames in, arrays out:
  no path, no store, no format.
- `src/pipelines/serving.py` — persisting that object and reading it back, plus
  the fixture index the service answers from. The loader checks the manifest's
  checksum *before* unpickling, because a truncated artefact raises at best and
  produces something plausible at worst, and compares the libraries that fitted
  it against the ones running now.
- `src/storage/predictions.py` — the served-prediction log. PostgreSQL, which
  `src/storage/base.py` has named as the store that would arrive with the first
  thing to persist since Milestone 3. Optional: with no DSN the service runs
  without a database, which is what CI and a clean checkout do.
- `scripts/build_model.py` and `make model` — fit on the whole history, write
  the artefact and a manifest. Joined `make reproduce` as its ninth stage, so
  the one command now ends at something servable rather than at a document.
- `Dockerfile`, `docker-compose.yml`, `requirements-api.txt` — a two-stage
  build, non-root, with a health check. `data/` and `models/` are mounted
  read-only rather than copied in: retraining is then a restart instead of a
  rebuild. The serving requirements are a subset with a reason written beside
  each omission.
- `SECURITY.md`, `docs/API.md`.
- Two CI invariants and a third job. `src` must not import `api`, and `api` must
  not reach past the pipeline layer into the ratings, features or ingestion —
  the one shortcut that would put an unprobed copy of the feature layer on the
  request path. The new job builds the image and curls `/health` against it,
  because deployment code that is only ever built by hand is broken on the
  morning it is needed.

### Changed

- `MATCHES_FILENAME` moved from `src/pipelines/ingest.py` to
  `src/ingestion/base.py`, which `ingest` re-exports. Reading one string meant
  importing the provider adapter, the registry, the cache and — through them —
  `requests`, so the serving process was loading the ingestion stack to learn a
  filename. Found by the image failing to start, not by review.
- The three boosted libraries are imported when their family is built rather
  than when `src/models/zoo.py` is. They are separate wheels of a few hundred
  megabytes each and the served blend contains one of them; at module scope,
  the serving image had to carry LightGBM and CatBoost to satisfy an `import`
  no request reaches.
- `SHIPPED` moved from `src/pipelines/report.py` to `src/models/ensemble.py`.
  The model layer names the model; the reporting layer and the API now read the
  name from one place instead of agreeing about a string.
- The model card's limitations are a constant, `LIMITATIONS`, that `render()`
  splices in — so the four bullets the API serves and the four the document
  states are the same object. The rendered card is byte-identical.
- Coverage, lint, format and type-check cover `api` as well as `src`; the
  threshold is still 100%.
- `ApiConfig` gained `max_batch` and `prediction_log_dsn`. The DSN is
  environment-only and has no line in `configs/config.yaml`, because that file
  is committed and a setting with nowhere to write it is one nobody commits by
  accident.

### Fixed

- The serving image honours `API_PORT`. It declared the variable, the
  `HEALTHCHECK` read it and the `CMD` hardcoded the port, so `-e API_PORT=9000`
  left the app serving on 8000 while the probe asked 9000 and failed every
  time — a container reported unhealthy forever while answering correctly. The
  command is now `sh -c` with an explicit `exec`, which expands the variable
  *and* keeps uvicorn as PID 1, so a stop is still half a second rather than
  the full grace period and a SIGKILL. `API_HOST` is dropped from the image
  rather than honoured: a container bound narrower than `0.0.0.0` is a mistake,
  and a configurable bind is one a loopback health probe can be configured out
  of.
- `pd.Timedelta(days=1)` now raises a `DeprecationWarning` under the installed
  numpy. Caught by the suite's `-W error::DeprecationWarning`, which is what it
  is for.

### Deviations from the plan, with reasons

- **No SQLAlchemy.** `requirements.txt` pencilled it in beside psycopg. One
  table, created if absent, with no column ever dropped or renamed: an ORM and
  a directory of versioned migrations would be a second description of a table
  that fits on a screen. psycopg alone, with every value bound.
- **The service does not price unplayed fixtures**, because the provider
  publishes no fixture list — see the note in `docs/API.md`. The scope said
  "one fixture in, three probabilities out", and what it can be handed is a
  fixture the batch build has a design row for.


Repository polish. No model, feature, evaluation or calibration behaviour
changes; every reported number is byte-identical.

### Added

- `make reproduce` — the eight stages from an empty checkout to
  `docs/MODEL_CARD.md`, in the only order they work in, with the runtime and
  the outputs of each documented in the README. Recursive `$(MAKE)` rather than
  prerequisites, because a prerequisite list is a set and `make -j` may run a
  set in any order.
- `uv.lock` is now committed. `requirements*.txt` pins the direct dependencies;
  the lockfile pins the transitive closure with hashes, which is the difference
  between reproducing the benchmark numbers and approximately reproducing them.

### Changed

- The coverage gate is `--cov-fail-under=100`, in both `make test-cov` and CI.
  The suite measures 100% and the README badge says 100%; a threshold five
  points below the claim let the claim rot without the gate noticing.
- `make ratings` is documented as ~10 min rather than ~20, matching the
  measurement in `docs/RATINGS.md` — sixty days between Dixon-Coles refits
  halved the build and the help string was never updated. `make ablation` is
  ~10 min in the README, matching the Makefile.

### Fixed

- The canonical table's check count is **24** everywhere. The README said
  twenty-three in two places and twenty-four in two others; `match_checks()`
  returns 24 and `docs/DATASET_CARD.md` reports 24.
- The competition count is **39** everywhere. The README strapline and the
  `pyproject.toml` description said ~38.
- Milestone references that a later milestone answered are stated as answered:
  the rest-days ablation (README, `docs/FEATURES.md`), the gap the model zoo
  was aimed at (README, `docs/EVALUATION.md`), the per-class breakdown now in
  `docs/MODEL_CARD.md` (`docs/EVALUATION.md`), and the two per-competition
  rating questions that Milestone 7's splits made answerable and that remain
  open (`docs/RATINGS.md`).

## [0.10.0] — unreleased

Milestone 10: evaluation and explainability. Three ways of asking what a
feature block is worth, and a card generated from the runs that measured the
model.

### Added

- `src/explainability/permutation.py` — break a block at prediction time and
  see what log loss loses. Model-agnostic, so it covers all six families, and
  reported in the same units as Milestone 8's ablation. A block is shuffled
  **jointly**, not column by column: permuting fourteen correlated form columns
  independently builds fixtures that never happened and measures the model on
  nonsense. One fit, thirty-one predictions — the estimator depends on the
  training half, which no permutation touches.
- `src/explainability/shapley.py` — `TreeExplainer` over the three boosted
  families, aggregated to the same five blocks the ablation used. Magnitude
  summed over classes, because SHAP contributions are signed per class and a
  signed average is approximately zero for the columns that matter most. The
  three pipeline-wrapped families are refused **by name**, with the method that
  does cover them in the message.
- `src/evaluation/model_card.py` and `docs/MODEL_CARD.md` — the sibling of the
  dataset card, and generated for the same reason. It renders from data it is
  handed and does not know that DuckDB, Parquet or a backtest exist, which is
  what keeps `src/evaluation` arithmetic over arrays and lets the card be
  tested against four hand-written frames.
- `src/pipelines/report.py` — the two breakdowns Milestone 9 left pooled: per
  class, and per competition. Both off the same per-match pass.
- `src/pipelines/tables.py` — the three tables joined once, for the three
  commands that need them. Lifted out of `scripts/train.py`, where the next
  copy was about to be made.
- `reliability(..., classes=("D",))` — bin one class alone. Pooling the three
  answers whether the model is honest; splitting them says which class it is
  dishonest about.
- `scripts/explain.py` (`make explain`), `scripts/model_card.py` (`make card`),
  **[docs/EXPLAINABILITY.md](docs/EXPLAINABILITY.md)**, and a CI invariant
  keeping `src/explainability` above the storage and pipeline layers.

### Measured

LightGBM on the most recent fold, 291,715 training matches and 11,802 scored:

| Block | breaking it | share of the arithmetic | never having had it |
|---|---:|---:|---:|
| `elo` | **+0.0395** | **38.0%** | +0.0021 |
| `form` | +0.0087 | 37.8% | **+0.0033** |
| `dixon_coles` | +0.0073 | 19.9% | +0.0016 |
| `schedule` | +0.0001 | 2.4% | +0.0003 |
| `head_to_head` | +0.0001 | 2.0% | +0.0001 |

- **The methods disagree about the top two, and the disagreement is the
  finding.** Permutation and SHAP rank elo above form; the ablation ranks form
  above elo. The model reaches for elo hardest and can most easily do without
  it, because Dixon-Coles is a substitute — the same fact Milestone 8 measured
  as "the two ratings are substitutes", now visible as a 19× gap between what a
  block is used for and what it is worth. Nothing substitutes for form, so its
  two numbers agree.
- **Where all three agree, they agree completely.** `schedule` and
  `head_to_head` sit at 0.0001–0.0003 by every method on every family. Three
  independent measurements at the noise floor settles what Milestone 5 left
  open.
- **Every family ranks the blocks the same way**, with the MLP leaning on the
  ratings twice as hard as anyone else — the same MLP whose errors correlate at
  0.92 with the rest of the zoo, and which is in the blend for that reason.
- **The shipped model is most honest about draws and least useful there.** Per
  class, the calibrated blend's calibration error is 0.0012 for draws against
  0.0035 for home wins — because it states about a quarter every time and about
  a quarter of matches are drawn. Reliable is not the same as useful, which is
  why the card carries the score table beside the reliability one.
- **Per competition, the Argentine cup is the least reliable of the 39** at
  0.0270 against a pooled 0.0015 — the same competition Dixon-Coles could not
  price in Milestone 7.

### Changed

- `TrainedForecaster` gained `fit()` and `predict()`, and `forecast()` is now
  the composition of the two. Behaviour is identical; SHAP needs the fitted
  estimator and permutation needs to predict nineteen times off one fit, and
  both would otherwise have reimplemented the class-scattering that makes a
  forecast come back in H/D/A order.
- `fold_forecasts` carries `competition_id`. The per-competition breakdown is a
  grouping the per-match frame can answer and the scored table cannot — the
  backtest keeps competition and fold, but only as means.
- `scripts/train.py` reads its tables through `src/pipelines/tables.py`. Same
  behaviour, one copy.
- `shap>=0.46` added to `requirements.txt`, with a `mypy` override — it ships
  no `py.typed`. **`matplotlib` was in the plan for this milestone and is not
  installed**: every figure it would have drawn is a five-row table.

### Notes

- The model card carries **no generation timestamp**, unlike the dataset card.
  The dataset card's subject is a download that changes underneath it; this
  one's is a version-controlled model, and a header that moves on every run
  turns "the numbers changed" into a diff nobody reads. There is a test that
  two runs produce the same bytes.
- SHAP covers three families of six, on purpose. `TreeExplainer` needs the
  estimator to *be* the tree ensemble, and the general-purpose explainer costs
  hours per family for a number permutation importance produces exactly, for
  every family, in seconds.
- The attribution runs on one fold rather than five. It is a ranking whose gaps
  are an order of magnitude apart, and a second fold does not move it.
- No per-match explanations. SHAP can decompose a single forecast; a plausible
  story about one fixture is the most misusable output this layer has, and the
  card says so in its own limitations.

## [0.9.0] — unreleased

Milestone 9: ensembling and calibration. Two layers over the zoo, and the
result is how little they are worth.

### Added

- `src/models/ensemble.py` — a mean of several forecasters, as one forecaster,
  whose members are chosen on **the correlation of their per-match errors**
  rather than on their individual scores. Measured on the tuning slice, the
  four tree-based families sit between 0.9934 and 0.9960 of each other,
  logistic regression at 0.9857 and the MLP alone at 0.92; the threshold sits
  in the empty band between 0.9857 and 0.9934 and admits XGBoost, logistic
  regression and the MLP. Averaging the top three instead would have averaged
  three near-substitutes and reported the averaging.
- `src/models/calibration.py` — temperature scaling, one scalar, fitted on the
  last year of each fold's own **training** half with the inner model refitted
  on everything before it. That costs a second fit per fold and is the only
  arrangement under which a calibrated model has seen exactly the matches the
  uncalibrated one saw.
- `src/evaluation/reliability.py` — does a stated probability happen as often
  as it says? All 3n statements a three-class forecast makes are binned, not
  only the model's favourite class: binning the favourite measures a
  classifier's confidence and says nothing about the draw column, which is most
  of what this model states.
- `run_ensemble`, `ensemble_forecasters` and `reliability_tables` in
  `src/pipelines/train.py`; `--ensemble`, `--member`, `--correlations` and
  `--bins` on `scripts/train.py`; `make ensemble` (~20 min) and
  `make correlations` (~3 min).
- Selection constants baked into `src/models/ensemble.py` the way the
  hyperparameters are baked into `zoo.py` — reviewed in a diff, re-derivable
  with `--correlations`.

### Measured

Five yearly folds, the same 59,001 matches every forecaster could price:

| | log loss | RPS | calibration error |
|---|---:|---:|---:|
| Bookmaker closing odds | **0.9993** | **0.2031** | — |
| Blend of three, calibrated | **1.01560** | **0.2082** | **0.0015** |
| Blend of three | 1.01565 | 0.2082 | 0.0045 |
| CatBoost, best single family | 1.01589 | 0.2083 | — |
| XGBoost, calibrated | 1.01622 | 0.2084 | 0.0020 |
| XGBoost | 1.01614 | 0.2084 | 0.0037 |

- **The blend beats every family that went into it, and the one that did not**
  — 0.0005 over its best member, 0.0002 over CatBoost. Per competition it beats
  XGBoost in 28 of 39, not 39.
- **Calibration halves the calibration error and does not move the log loss.**
  On XGBoost the loss is 0.00008 worse; on the blend, 0.00005 better. Both are
  noise. That is the answer to the question rather than a disappointment: log
  loss is a proper scoring rule and these models were fitted on it, so what was
  left was a small overconfidence the score barely charges for.
- Together the two layers close **0.0003** of the 0.0165 Milestone 8 left,
  leaving 0.0163 to the closing line.

### Changed

- `docs/LEAKAGE.md` gains the two boundaries this milestone adds: the
  calibration holdout, cut inside the training half on the date, and the member
  selection, run on the tuning slice. Both are places where something is
  *chosen*, which is a way for the evaluation half to reach a model without any
  column moving.
- `docs/MODELS.md` gains the correlation matrix, the per-fold temperatures and
  the reliability table; `docs/EVALUATION.md`'s "not measured yet: calibration
  curves" is now measured.
- `src/models/ensemble.BEST` names XGBoost, which leads the **tuning slice**.
  CatBoost leads the reported folds by 0.0002 — smaller than the difference
  Milestone 8 called indistinguishable — and picking the subject of a
  measurement by looking at the folds it is measured on is the mistake the
  member selection is arranged to avoid.

### Notes

- No fitted ensemble weights and no stacking. A meta-model over three
  correlated members is a fourth thing to tune and validate for a margin
  already at 0.0005; the plain mean is the version whose number can be
  attributed to the members.
- No vector or matrix scaling. Three or twelve parameters instead of one, for a
  reliability table whose largest bins are already within 0.0001 after the
  scalar.
- `make ensemble` walks the folds twice on purpose — once through the unchanged
  backtest for the scores, once more for the reliability tables. The backtest
  persists means, and reliability is a question about individual probabilities;
  the alternative is a side channel out of a scoring pipeline whose ignorance
  of what it scores is why a model and a baseline can share a table.

## [0.8.0] — unreleased

Milestone 8: the model zoo. Six families, and a result that is more interesting
than the ranking.

### Added

- `src/models/zoo.py` — logistic regression, random forest, XGBoost, LightGBM,
  CatBoost and an MLP, behind one wrapper. A trained model is a `Forecaster`
  that fits inside its own `forecast(train, evaluate)`, so **the evaluation
  pipeline from Milestone 7 is used unchanged** — nothing in
  `src/pipelines/backtest.py` knows an estimator exists, which is the only
  reason a model and a baseline can be put in one table.
- `src/models/dataset.py` — the thirty columns a model sees, **derived** from
  the feature and ratings registries rather than listed again. A feature added
  in Milestone 5 is a column the zoo sees without anyone editing a second list.
  Nothing canonical passes through directly: the scoreline and the odds are
  both excluded by a set intersection in the test suite, not by a rule someone
  has to remember.
- `src/models/tuning.py` — Optuna over a **tuning slice**: every match strictly
  earlier than the first reported fold, 241,481 of them ending 2021-09-02.
  Choosing settings by looking at the folds they are then scored on is the same
  mistake as fitting a rating on the matches it prices, and this project has
  already built and removed one thing for it.
- `src/models/tracking.py` — MLflow to a local SQLite file under `models/`. A
  tracking failure is logged and swallowed: the real output of a training run
  is the score table, and a command that failed because a log could not be
  written would be failing for an unrelated reason.
- `src/pipelines/train.py`, `scripts/train.py`, `make train` (~8 minutes) and
  `make ablation` (~5 minutes), plus **[docs/MODELS.md](docs/MODELS.md)**.
- `DuckDBStore.read_features()` and a `features=` view, matching the ratings.
- A CI note extending the `src/models` invariant to `feature_engineering.registry`,
  which is a schema module on the same terms as `ingestion.base`.

### Measured

Five yearly folds, 62,036 evaluation matches, scored on the 59,001 every
forecaster could price:

| | log loss | RPS | accuracy |
|---|---:|---:|---:|
| Bookmaker closing odds | **0.9993** | **0.2031** | 50.6% |
| CatBoost | **1.0159** | 0.2083 | 49.3% |
| XGBoost | 1.0161 | 0.2084 | 49.3% |
| LightGBM | 1.0161 | 0.2083 | 49.3% |
| Logistic regression | 1.0162 | 0.2083 | 49.3% |
| Random forest | 1.0172 | 0.2087 | 49.2% |
| MLP | 1.0203 | 0.2091 | 49.1% |
| Dixon-Coles | 1.0277 | 0.2114 | 48.5% |
| Class prior | 1.0751 | 0.2284 | 43.7% |

**Every family beats the rating, in all 39 competitions.** The best closes
0.0118 of the 0.0284 Milestone 7 measured against the closing line — 42% — and
leaves 0.0166.

**The top four are within 0.0003 of each other.** Logistic regression on thirty
columns is not distinguishable from three tuned gradient-boosting libraries.
Milestone 7 found that what the bookmaker knows on top of a strength rating is
not strength; this adds that it is not a non-linear function of these thirty
columns either. What remains is missing information, not missing capacity — and
that is a finding about where Milestone 9 should not look.

The ablation, LightGBM with each block withheld, all variants in one backtest:

| Block withheld | Δ log loss |
|---|---:|
| `form` | **+0.0033** |
| `elo` | +0.0021 |
| `dixon_coles` | +0.0016 |
| `schedule` | +0.0003 |
| `head_to_head` | +0.0001 |

**Form is worth more than either rating** — fourteen rolling windows against a
bivariate Poisson refitted every sixty days. The two ratings are substitutes,
so each looks small alone. **Rest days and congestion are worth 0.0003**, which
settles the question Milestone 5 left open when it shipped them: they survive,
and they are the first thing to go if the feature set needs trimming.
**Head-to-head is worth 0.0001**, which is nothing.

**The zoo fixes the competition the rating could not price.** Dixon-Coles lost
to counting base rates on the Argentine cup, 1.1329 against 1.0902 — the one
competition of 39 where that happened. LightGBM gets 1.0806 there, without a
special case or a per-competition rule. Milestone 7 asked whether the cup
needed a pooled Dixon-Coles fit; the answer is that it needed a model around
the rating, not a change to it.

Tuning bought little. Four of the six searches moved the fourth decimal place,
and twenty trials could not beat `C=1.0` for logistic regression at all — with
241,000 rows and thirty columns the penalty is irrelevant. The two that did
move were bad starting guesses rather than subtle optima: LightGBM's defaults
were too coarse (+0.0028), and the MLP's were wrong (+0.0144 for one hidden
layer instead of two), which is a fair measure of how wrong an untested guess
at an architecture can be — and it is still the worst family afterwards.

### Changed

- **MLflow writes to SQLite, not to its own directory store.** The file store
  is in maintenance mode in MLflow 3.x and raises on write unless an
  environment variable opts out of the warning. The experiment's artefact root
  is also set explicitly, because the default is `./mlruns` relative to the
  working directory — which the clean-checkout gate would then fail on.
- **The tree search spaces were narrowed once, after measurement.** LightGBM's
  first version reached 255 leaves and 800 estimators, and one trial in that
  corner took longer than the entire XGBoost search, for a model nobody would
  ship on thirty tabular columns. Every reported search used the narrowed space.
- **The seed fixes the model, not the last bits.** Every family fits on all
  cores and a parallel sum reorders floating-point addition, so two runs of the
  forest agree to about 1e-9 rather than bit for bit. Single-threading would
  buy exact reproducibility for roughly four times the runtime, and the
  smallest difference read off an ablation here is 1e-4.

## [0.7.0] — unreleased

Milestone 7: splits and baselines. The first milestone that produces a number
the whole project is aimed at, and it is not a flattering one.

### Added

- `src/models/splits.py` — walk-forward folds. Expanding training window, five
  folds of a year each, anchored at the end of the history. **The cut is a
  date, never a row**: a boundary at row *n* puts two matches played on the
  same afternoon on opposite sides of it, so the model trains on the 3pm
  results and is scored on the 5.30 kick-off. Every fold runs through
  `split_boundary` before it is yielded, and a violation raises.
- `src/models/baselines.py` — four forecasters. Home-always, the class prior
  counted on the training half, Dixon-Coles from the ratings table, and the
  closing line with the overround removed. Each returns null for a match it
  cannot price; nothing invents a number to fill a gap.
- `src/evaluation/metrics.py` — log loss, RPS and accuracy, in that order of
  importance. Log loss is **not clipped**: a forecast that ruled out what
  happened scores infinity, and reporting that as a large finite number would
  be a kindness the metric does not extend. RPS is what still ranks such a
  forecast, because it is bounded and knows H, D and A are ordered.
- `src/pipelines/backtest.py`, `scripts/backtest.py`, `make backtest` (~5
  seconds) and **[docs/EVALUATION.md](docs/EVALUATION.md)**.
- `PathsConfig.reports_dir` and `data/reports/`. A backtest result is evidence
  about a particular set of forecasters and outlives them; a feature table is
  rebuilt whenever the feature set changes. Different lifetimes, different
  directories.
- `DuckDBStore.read_ratings()`, unfiltered on purpose: the caller joins to the
  slice it already holds, and a second filter API is a second place for a
  point-in-time read to be subtly different.
- Two CI invariants: `src/evaluation` may import the canonical result labels
  and `src/utils`, and `src/models` may import schemas, the metrics, the probes
  and `src/utils`. A model that went to disk for the rest of the history would
  pass every fold check while training on the future.

### Measured

Five yearly folds, 62,036 evaluation matches, scored on the 59,001 every
forecaster could price:

| | log loss | RPS | accuracy |
|---|---:|---:|---:|
| Bookmaker closing odds, overround removed | **0.9993** | **0.2031** | 50.6% |
| Dixon-Coles | 1.0277 | 0.2114 | 48.5% |
| Class prior, counted per fold | 1.0751 | 0.2284 | 43.7% |
| Home always | ∞ | 0.4316 | 43.7% |

**0.0284 of log loss** between the rating and the closing line. That is the
honest size of what the model zoo has to close, and it is measured on identical
matches rather than on two convenient subsets — the bookmaker quotes 99.8% of
these matches and Dixon-Coles prices 95.3%, which are not the same matches.

Fold to fold the rating moves over 0.008 and the line over 0.008, with the gap
never leaving 0.021–0.028. Nothing here is a lucky year.

Two findings, both pinned by integration tests:

- **The rating's edge over the prior tracks the spread of team strength at
  r = 0.90** across the 39 competitions. A strength model has the most to say
  where strengths differ most, which is what it should do and is not something
  anyone told it to do.
- **The gap to the closing line tracks that same spread at −0.14** — that is,
  not at all. Whatever the bookmaker knows on top of the rating is not
  strength, and it is worth about the same amount in every competition. That is
  the target for Milestone 8, stated as a number before any feature is chosen.

**Dixon-Coles loses to counting base rates in one competition of 39**: the
Argentine cup, 1.1329 against 1.0902. The obvious explanation is wrong — all 32
of its recent clubs also play in ARG_1, and its Elo spread is the seventh
*narrowest* of the 39. The sides are closely matched, so a strength model has
little to add, and the model refits per competition, so the cup's fit sees 610
matches while the same clubs' 2,211 league matches sit next door unused. Left
in rather than special-cased; what it argues for is pooling a cup's fit with
its country's league, which is a change to the rating and needs an ablation to
justify it.

### Changed from the approved scope

- **The metrics live in `src/evaluation/`, which the plan gave to Milestone
  10.** They are needed to report a baseline at all, and writing them in
  `src/models` to move them later is churn with a rename in it. The same
  precedent as Milestone 4 writing the temporal probes that Milestone 6 owned.
- **Home-always is reported, and the class prior is what it is compared
  against.** The scope named home-always as the baseline to beat. It is not a
  probability forecast anybody should be scored against — its log loss is
  infinite — so it is reported for what it does show (43.7% accuracy, and what
  a proper scoring rule does to certainty) and the honest floor is the class
  prior counted per fold.

## [0.6.0] — unreleased

Milestone 6: the leakage suite. Two more probes, and a change of principle —
the suite no longer takes a list of what to check.

### Added

- `src/validation/leakage.py`. It **walks** `src.ratings` and
  `src.feature_engineering` and finds every class satisfying the producer
  contract, constructed with the defaults the pipelines use. Both earlier
  milestones probe the producers on their own list, and a list is a thing you
  can forget to add to: a builder wired into a pipeline but omitted from its
  probe call would have shipped unverified, with nothing in the output saying
  so. `check_defaults_are_complete()` closes the other half — a discovered
  producer in no pipeline's defaults is a column the model layer expects and
  will not get.
- **Split boundary**, the third probe. No training row may be dated at or
  after any evaluation row, and no match may appear in both halves. Ties fail:
  a full Saturday programme is one round, and a model trained on the 3pm
  results is not entitled to predict the 5.30 kick-off. Milestone 7 owns the
  splits; the probe is here and tested, waiting for them.
- **Observed reads**, the fourth. Rewrite one input column, recompute, and
  whatever moved read it. This is the measured counterpart to the feature
  registry's hand-written `reads` — the one part of that registry that can be
  wrong without anything noticing, since a typo there reclassifies a leaking
  feature as safe. The suite asserts the declaration *covers* what was
  measured; over-declaring is safe, under-declaring is not.
- `benchmark_leaks()`: any derived column that moves when a bookmaker's price
  is rewritten fails the run. The rule was written on `ODDS_COLUMNS` in
  Milestone 2 and until now was only written.
- `tests/unit/test_leakage_suite.py` — both temporal probes over every
  discovered producer, on every test run. A unit test rather than a shell step
  on purpose: CI runs the suite, so a producer added on a branch is probed by
  the same command its author already runs locally. CI names it as its own
  step so a leak appears in the job list rather than as one dot among five
  hundred.
- `scripts/audit_columns.py`, `make audit`, and **[docs/LEAKAGE.md](docs/LEAKAGE.md)**:
  every derived column the model layer will see, traced back to the canonical
  columns it reads.

### Measured

Thirty derived columns traced over the most recent 1,200 matches of ENG_1 and
ESP_1. Every probe held; nothing reads `result`, `ht_*`, `referee` or an odds
column. Four things the trace says that reading the code does not:

| Finding | Why it matters |
|---|---|
| Elo reads `season`, never `date` | Its ratings are invariant to *when* matches were played and sensitive only to order — correct for an online update, and worth knowing before anyone adds time decay to it. |
| Dixon-Coles reads `competition_id`; Elo reads no partition column at all | Elo's per-country pooling is already inside `home_team_id`, which is country-scoped so a promoted club keeps one id. The partition is in the identifier, not in a `groupby`. |
| `home_venue_points_5` does not read `away_team_id` | Venue form is a team's record at its own end and the opponent is irrelevant to it. This is the asymmetry a reshape is most likely to get wrong, and it shows up as an absence rather than as a number to check by eye. |
| `elo_*_played` reads no result | The same conclusion the registry reaches by declaration for `*_matches_played`, reached here by measurement. |

The trace is a **lower bound**, and both ways it falls short are properties of
the sample rather than of the code. A column that is entirely null cannot be
perturbed: run the audit over ENG_1's first 2,000 matches and it reports that
no feature reads shots — correctly, for that slice, since the provider carried
none before 2000/01. A column that does not vary cannot be perturbed either, so
a single-competition sample cannot show that Dixon-Coles partitions on
`competition_id`. Hence the most recent matches, from two competitions in two
countries.

### Changed

- `--limit` on the audit takes the most **recent** N matches per competition
  rather than the first. The first N are the ones with no shot data, and an
  audit over them reports absences that belong to 1993 rather than to the code.

## [0.5.0] — unreleased

Milestone 5: feature engineering. Twenty features, and the correctness bug that
building them uncovered.

### Added

- `src/feature_engineering/` — 20 features in three groups. Form, venue form,
  goals and shots for and against, rest days, fortnight congestion, and
  head-to-head record.
- `windows.py`, the causal primitive: every window ends at the last row
  **strictly earlier by date**. `shift(1).rolling(k)` counts rows, so two
  matches on the same date let the earlier one — earlier only by an arbitrary
  tiebreak — inform the later.
- `registry.py`, where each feature declares the canonical columns it reads.
  Whether it *can* leak is derived from that rather than declared separately,
  so there is no field to set wrongly. Seven of the twenty read nothing but the
  fixture list.
- `src/validation/features.py` — 10 checks, including a table-level leakage
  gate: every window feature has a companion count, and a value where the count
  is zero came from somewhere it should not have. It runs over every row
  written, where the probes see one sampled competition.
- `src/pipelines/features.py`, `scripts/build_features.py`, `make features`
  (~10 seconds) and `make feature-list`.
- A CI invariant: `src/feature_engineering` may import the canonical schema and
  `src/utils`, nothing else. A builder that reached for the store could pass
  every probe while reading the future, because the probes work by handing it
  truncated frames.

### Changed

- **`src/pipelines/derived.py` extracted.** The ratings and feature pipelines
  had the same shape — run producers, probe, check, persist, report — and
  written twice the two would drift, with the drifting half always being the
  probe nobody looked at again. The ratings pipeline moved onto it with its
  tests unchanged.
- Feature checks are scoped to what a build claims to have produced, so
  `--builder head_to_head` is not reported as twelve null columns. A warning
  that fires on a legitimate command is one people learn to ignore.

### Measured

Home win rate by the gap between the two sides' five-match form:

| Form gap | Matches | Home | Draw | Away |
|---|---:|---:|---:|---:|
| below −1.2 | 38,186 | 31.7% | 27.2% | 41.1% |
| −0.2 to 0.2 | 61,169 | 45.1% | 27.5% | 27.3% |
| above 1.2 | 26,201 | 61.1% | 22.2% | 16.7% |

A thirty-point spread from one feature — and the draw rate peaks between evenly
matched sides and falls at both extremes, which football says should happen and
the feature was not built to produce.

Venue form is worth ±0.275 points per game, symmetric between the two sides to
three decimals: a home team takes 1.618 at home against 1.343 overall, an away
team 1.119 away against 1.392.

**Rest days carry no marginal signal**: 43.7% to 45.6% home wins across the
range, and not even monotone. Kept for the interaction rather than the
marginal; Milestone 8's ablation is where it earns its place or is dropped.

### Fixed

- **Five of the provider's early files are copies of `SP1.csv` served under
  another name**, and the adapter believed the URL. `1993-94/P1.csv`,
  `1993-94/SC1.csv` and `SP2.csv` for 1993-94, 1994-95 and 1995-96 all carry
  Spanish La Liga rows — with `Div` inside each file saying `SP1`. The result
  was 380 fabricated Portuguese matches and 380 fabricated Scottish ones,
  Spanish clubs wearing `por:` and `sco:` team ids, plus 1,222 La Liga
  fixtures duplicated into Spain's second division.

  Every copy was internally consistent — same teams, same date, same score —
  so no per-row check could see it, and `match_id` hashes the competition, so
  deduplication kept both. The adapter now trusts each file's own `Div` column
  and drops rows that name a different division; a new validation check,
  **"no fixture appears in two competitions"**, is what says it worked.

  Found by an assumption check written for the feature layer: rolling form
  needs a team never to play twice on one date, and 2,444 team-days said
  otherwise.

### Changed

- The ingest is **303,517 matches and 1,323 teams**, down from 305,499 and
  1,363. Portugal's first league season now starts 1994-08-20 and Scotland's
  second tier 1994-08-13, rather than both starting in Spain in September 1993.
- Every measured figure moved in the fourth decimal. Elo's error over the
  corrected table is 0.16177 against 0.16169 before; the Premier League
  Dixon-Coles numbers are unchanged, because England was never affected.

## [0.4.0] — 2026-09-03

Milestone 4: ratings. Two of them, and the machinery that proves neither can
see the match it is rating.

### Added

- `src/ratings/elo.py` — one Elo pool per country, online, with home advantage,
  a margin-of-victory multiplier, autocorrelation damping and season
  carry-over. Online is the point: the rating carried into match *n* is a
  function of matches 1..*n*-1 by construction.
- `src/ratings/dixon_coles.py` — bivariate Poisson with the low-score
  correction and exponential time decay, refitted per competition on a rolling
  window that ends **strictly before** the match that triggered the refit. It
  produces a real H/D/A distribution, so it is the first thing here that can be
  scored against a bookmaker.
- `src/validation/temporal.py` — prefix invariance and outcome independence,
  generic over `Callable[[DataFrame], DataFrame]`. Scoped for Milestone 6 and
  written here, because shipping two rating models with their central property
  unverified for two milestones is not a trade worth making.
- `src/validation/ratings.py` — 9 arithmetic and coverage checks on the output.
- `src/pipelines/ratings.py` and `scripts/build_ratings.py`. The build runs the
  probes against a sample competition and exits non-zero when one fails.
- `make ratings` (~10 min) and `make ratings-elo` (seconds).
- A CI invariant: `src/ratings` depends only on `src.ratings` and `src.utils`.
  A model that reached for the store could pass every probe while reading the
  future, because the probes work by handing it truncated frames.

### Measured

Elo, over all 305,499 matches, every prediction from prior matches only —
mean squared error of the expected score:

| | MSE |
|---|---|
| Plain Elo | 0.16882 |
| + home advantage | 0.16217 |
| + margin of victory | 0.16208 |
| + autocorrelation damping | 0.16173 |
| + season carry-over (shipped) | **0.16169** |

Dixon-Coles on the English Premier League. All three computed over exactly the
same 8,818 matches — the ones the model priced *and* the provider carries a
closing line for:

| | log loss | RPS |
|---|---|---|
| Class prior | 1.0636 | 0.2285 |
| **Dixon-Coles** | **0.9774** | **0.1988** |
| Bookmaker closing odds, overround removed | 0.9619 | 0.1941 |

Which is where the README says a public-data model should land, from a rating
rather than from the model zoo. Over all 12,052 matches it could price,
including three decades before the odds series starts, log loss is 0.9888.

The full build, measured: 305,499 matches rated in about ten minutes,
Dixon-Coles pricing 92.1% of them and Elo all of them, all four causality
probes holding, and 9 of 9 checks passing with none skipped.

### Changed

- Elo's `season_carry` ships at 0.97, not the football-Elo convention of 0.75.
  Three decades of results say a club's strength persists across a summer far
  more than the convention assumes.
- Dixon-Coles ships with a 347-day decay half-life and a three-season window,
  both longer than the literature's, and refits every 60 days rather than 30 —
  which measured no worse and halves the runtime.

### Removed before shipping

- **Per-competition Elo fitting**, which the plan called for. Built, measured,
  removed: it made the ratings worse (0.16300 against 0.16246), because a
  calibration window is the coldest part of the history and three free
  parameters chase warm-up noise. `--fit-until` re-derives the constants as a
  maintenance step.

### Fixed

- `DuckDBStore.open_matches` attached the matches view, failed on a missing
  ratings file, and left a file-backed catalog holding half of what was asked
  for — which the next open would have reported as success. Every source path
  is checked before the catalog is touched.

## [0.3.0] — 2026-09-03

Milestone 3: storage and validation. The canonical table becomes queryable
through an interface, and the assertions that lived in a test file become a
report the pipeline runs on every ingest.

### Added

- `src/storage/` — a `MatchStore` protocol and a DuckDB implementation over the
  Parquet the ingest pipeline writes. Views rather than tables, so the file
  stays the single source of truth; filters (`competitions`, `since`, `until`,
  `columns`) are pushed into the query, which makes a point-in-time read the
  interface's own operation rather than something every caller reimplements.
- `src/validation/` — 23 checks in four groups (schema, integrity, referential,
  distribution) producing a report with three outcomes and two severities. Run
  by `run_ingest` on the frame it just wrote, and by
  `scripts/validate_data.py`, which exits non-zero on a blocking failure.
- `PRE_MATCH_COLUMNS` / `POST_MATCH_COLUMNS` / `BENCHMARK_COLUMNS` on the
  canonical schema, and a check that every column is classified. The leakage
  defence has to be temporal, not columnar: `home_shots` is kept because a
  team's shots in *earlier* matches are a legitimate feature, so it cannot be
  enforced by leaving data out.
- `docs/DATASET_CARD.md`, generated from the table on every validation run,
  with per-competition coverage and the checksum of the exact file it
  describes.
- `make validate`, `make validate-strict`.
- A CI invariant: the canonical table is read through `src/storage`, nowhere
  else.

### Changed

- `IngestReport` carries a `validation` report. It reports rather than gates —
  a table you can inspect is more useful than one the pipeline refused to save.
- `tests/integration/test_real_provider_files.py` no longer restates the data
  assertions. They live in `src/validation` and the integration suite runs
  them, rather than being the same rule in two places.

### Found

- **One row in 305,499 is filed under the wrong season** — an Argentinian match
  played 2015-01-29 and labelled 2013-14 in the provider's own file, 213 days
  outside the widest defensible window. Reported as a warning, left in place.
  The first thing the new gate caught, on the data that already existed.
- **`_goals_are_plausible` crashed on an empty table.** `min()` over an empty
  nullable column returns `pd.NA`, and `pd.NA < 0` raises rather than being
  falsy — so a validation suite would have died at the moment a report was most
  useful. Caught by the empty-table test, not by review.
- **DuckDB's dtypes depend on the rows selected**: a NumPy `int16` for a column
  with no nulls and a nullable `Int16` for one with them. Every store read
  re-asserts the canonical schema, which is also what makes a store read equal
  a `pd.read_parquet` of the same file.

### Deferred, with a reason

- **PostgreSQL and `docker-compose.yml`.** Scoped for this milestone, and
  moved to Milestone 11. There is no application data and no prediction to
  serve yet, so a Postgres store today would be an unused implementation of an
  interface with one caller, plus a compose file nobody runs — the dead code
  the project's own standing rule says to avoid. The seam that makes it cheap
  (`MatchStore`) is delivered.
- **MLflow.** Arrives with the model zoo in Milestone 8, when there is a run to
  track.
- **pandera.** The checks here are mostly semantic — home advantage, a book's
  overround, a season's date window — not schema, and pandera's schema model
  would be a second copy of `CANONICAL_SCHEMA` to keep in step.

## [0.2.1] — 2026-09-03

Incremental, idempotent re-runs. Verifying the claim that a re-run is cheap
found that it was not: settled files were trusted forever so a provider
correction was permanently invisible, live files re-downloaded on age even when
unchanged, unavailable seasons were re-probed every run, and nothing recorded
the checksum of a single input file.

### Added

- HTTP conditional requests (`If-None-Match` / `If-Modified-Since`). The
  provider answers both with 304 and no body, so change detection is
  authoritative instead of guessed. `HttpClient.download` now returns a
  `DownloadResult` reporting whether anything was transferred.
- `src/ingestion/cache.py` — per-URL `ETag`, `Last-Modified`, `sha256` and
  last-checked time, plus recorded misses with a 7-day TTL so unpublished
  seasons are not re-probed every run. A corrupt or version-mismatched cache is
  discarded rather than raised on.
- Raw-file manifest (`data/raw/manifest.json`) checksumming all 732 cached
  provider files — provenance of the inputs, alongside the existing manifest
  for the output.
- `--revalidate` re-checks finished seasons to pick up a provider correction;
  `--forget-misses` retries seasons previously recorded as unpublished.
  `make refresh` and `make revalidate`.
- 36 tests covering incremental re-runs, conditional requests and the cache,
  plus integration tests asserting the real manifest still verifies.

### Changed

- A secondary-feed country file is now revalidated **once per run** rather than
  once per season. Ingesting Brazil's fifteen seasons asked the server about
  the same file sixteen times.
- `max_age_days` removed. Age was the wrong question in both directions.

### Measured

Two full 39-competition ingests back to back: the second transferred **0 files
and 0.00 MB**, re-probed **0** of the previous run's misses, and produced a
**byte-identical** table — `sha256 8d489ed3…`, unchanged from before this work,
so the fetching got smarter and the data did not move.

## [0.2.0] — 2026-09-03

Milestone 2: ingestion.

### Added

- Provider-adapter layer (`src/ingestion/`). 39 competitions across 27
  countries and two provider file layouts reduce to one canonical 35-column
  schema; a full ingest yields 305,499 matches over 1993-2026. CI enforces that no module outside `src/ingestion` names the
  provider, so adding a source cannot leak into downstream code.
- Competition registry (`configs/leagues.yaml`). Adding a league is a config
  entry; `league_filter` splits country files that mix a league with a cup
  (Argentina) or two tiers (Switzerland).
- Season-label normalisation covering both split (`2024-25`) and calendar
  (`2024`) seasons, read per row — Japan's J1 League switches between them.
- Robust CSV reading: `utf-8-sig`/`cp1252` fallback, BOM handling, blank-row
  and ragged-row repair, and an HTML-served-as-CSV guard.
- Canonical team identity scoped to country (`eng:man-united`), stable across
  promotion and relegation, with a rename detector.
- Checksum manifests (`src/ingestion/manifest.py`) as dataset versioning.
- `scripts/fetch_data.py`, `make data`, `make leagues`, `make test-int`.
- `docs/DATA_SOURCES.md` — thirteen measured provider quirks, each naming the
  file it was found in.
- 16 integration tests asserting football-shaped invariants on real data
  (home-win rate, half-time ≤ full-time, shots on target ≤ shots, odds implying
  an overround above 100%). All skip without downloaded data.

### Changed

- Ragged-row handling now warns instead of raising. The original rule — surplus
  non-empty means a shifted row — was falsified by Italian Serie B 2003/04,
  whose 42-column header has eight blank names and whose 49-field rows are
  correctly aligned. Raising also aborted the entire run on one bad file out of
  roughly seven hundred.
- `PathsConfig.sample_dir` removed. No data slice is committed: the provider
  publishes no redistribution licence, and the unit suite is already offline.
- Every config section is now optional, since every field within one has a
  default.

### Notes

- No fuzzy team-name matching, and that was a measurement: 51 distinct team
  strings across 34 Premier League seasons, with the two most similar distinct
  pairs at exactly 0.800 similarity. Any threshold that catches a rename merges
  Sheffield United with Sheffield Wednesday.

## [0.1.0] — 2026-09-03

Milestone 1: repository foundation.

### Added

- Typed, frozen configuration (`src/utils/config.py`): YAML defaults overlaid
  by an explicit map of environment variables, validated by Pydantic v2.
  Unknown keys are an error rather than being silently ignored.
- Project-root-anchored path resolution (`src/utils/paths.py`), so no module
  computes a location relative to its own file.
- Idempotent logging setup (`src/utils/logging.py`) writing to stderr.
- A polite retrying HTTP client (`src/utils/http.py`) with atomic downloads:
  a truncated file is deleted rather than left to be parsed as a short season.
- Quality gates: ruff, black, mypy (strict), pytest. 71 tests, 100% coverage
  of `src`.
- `commit-msg` authorship hook, installed via version-controlled
  `core.hooksPath` so it survives a reclone.
- GitHub Actions CI running the same gates a developer runs locally.

### Notes

- `requirements.txt` grows one milestone at a time. A pinned dependency that
  nothing imports is a supply-chain surface with no upside.
- Pydantic does not run field validators on defaults unless
  `validate_default=True`. Without it, `data_dir` resolved to an absolute path
  when named in the config and a relative one when omitted — every downstream
  path would then have depended on the process's working directory. Fixed at
  the base model so no future section can reintroduce it.

[Unreleased]: https://github.com/Vanshcloud/Match-Outcome-Predictor/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/Vanshcloud/Match-Outcome-Predictor/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Vanshcloud/Match-Outcome-Predictor/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Vanshcloud/Match-Outcome-Predictor/releases/tag/v0.1.0
