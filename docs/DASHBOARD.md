# The dashboard

Milestone 12, with the fixture feed of Milestone 13 behind it. A football
dashboard over a prediction engine it is only ever a client of.

```bash
make dashboard      # http://127.0.0.1:8501
make api            # in another shell, for the forecasts
```

Or the whole stack, with the prediction log behind it:

```bash
docker compose up   # dashboard :8501, API :8000, PostgreSQL
```

---

## The one design decision worth reading first

**It computes nothing.**

Every measurement of the model is produced by `src/pipelines/report.py` or
`src/pipelines/backtest.py`. Every probability is answered by the service over
HTTP. Every result is read from the canonical match table. The dashboard
chooses *which rows* and *which figure*, and nothing else.

That is not tidiness. `docs/MODEL_CARD.md` is generated from the same
functions, so a number here and the same number in the card are the same number
from the same code — and the day they disagreed, nothing would be comparing
them. A dashboard that recomputed a calibration error would be a second
measurement of the model with no test holding it to the first.

The same reasoning, one layer down, is why forecasts come over HTTP rather than
from `models/servable.joblib`. Two processes that both unpickle the artefact
are two implementations of "what does the model say". CI enforces it:
`dashboard` may not import `api`, and the dashboard image does not contain it.

## Four layers, pointing one way

```
views  ──▶  services  ──▶  providers  ──▶  domain
```

| | What lives there | Why it is separate |
|---|---|---|
| `domain/` | `Fixture`, `Prediction`, `MatchStatus`, the competition catalogue, favourites | Value types with no I/O. A provider, a test and one day a React client all mean the same thing by a `Fixture`. |
| `providers/` | One protocol per kind of source, one implementation per source | The layer the next eight milestones plug into. A view never learns which provider answered. |
| `services/` | Orchestration and caching | What the home page *needs*, assembled from three providers and a favourites list. Streamlit reruns the whole script on every interaction, so the caching is real work. |
| `views/` | Streamlit | Thin, and the only layer that would be rewritten if this became a React client reading the same API. |

CI asserts the arrows: `domain` imports none of the three above it,
`providers` reaches up into neither, and **no view names a provider**. That
last one is the property the whole layer exists for — the day a view imports
`providers.historical`, connecting a real fixture feed stops being one class
and becomes a search.

## Three data paths, and why they differ

| | Comes from | Because |
|---|---|---|
| Results, form, head-to-head | `data/processed/matches.parquet`, through the storage layer | Results are what the provider actually publishes. The API deliberately serves no scoreline. |
| Scoreboard, reliability, per-competition | `data/reports/ensemble/*.parquet` | A measurement that already exists. Reading a file is the honest way to read one, and it works with no service running. |
| Forecasts | `POST /predict` on the running service | A live probability is the model's, and there should be exactly one process that holds the model. |

The consequence is deliberate: **the page works with the API down, and with no
data at all.** Each section says which of the three is missing and names the
command that produces it.

## The fixture feed

Three things need a source this project does not ingest: **today's matches,
upcoming fixtures and live scores**. The provider behind this repository
publishes *results* — a match that has not been played is in no table here — so
they come from a second feed or from nowhere, and both are real
implementations of the same interface.

| `DASHBOARD_FIXTURE_PROVIDER` | Class | What it answers |
|---|---|---|
| `none` (default) | `providers/null.py` | Nothing, and the reason. Every empty state on the page is *its* answer, not a special case in a view. |
| `football-data.org` | `providers/football_data_org.py` | `GET /v4/matches`: scheduled fixtures, matches in play with their score, and club crests. Nine competitions, needs `FOOTBALL_DATA_API_KEY`. |

```bash
# free key: https://www.football-data.org/client/register
export FOOTBALL_DATA_API_KEY=...
export DASHBOARD_FIXTURE_PROVIDER=football-data.org
make dashboard
```

**Nothing invents a fixture.** With no key, the two forward sections stay empty
and name the variable to set. Plausible-looking generated matches would put a
game on the screen that is not being played, and that is the one failure this
application cannot recover from: a reader who catches it once stops believing
the real rows too.

### What Milestone 13 actually cost

One class, one registry entry, one variable — the claim Milestone 12's shape
was making, now spent:

```python
FIXTURE_PROVIDERS = {"none": NullFixtures, "football-data.org": FootballDataOrgFixtures}
```

No view moved and no card changed. The one line of `views/` that did change was
the sidebar caption, which used to read `✕ No fixture feed — Milestone 13` and
now names the connected feed or the provider's own reason — a status bar citing
an unshipped milestone after it ships is a small lie a reader stops checking
the rest of the page against.

### What it is not

- **Not a second ingestion source.** Nothing this feed returns is written to a
  table, joined to one, or read by a model. Its rows live for one page render;
  when a match is played, its canonical row arrives from `make data` like every
  other result.
- **Not joinable to the match table, and it says so.** Ids are prefixed
  `fdorg-` rather than built with `make_match_id`, because a match id here
  hashes the team names *as the source spells them* and this feed says
  "Manchester United FC" where the ingested table says "Man United". An id that
  looked canonical and matched nothing would be worse than one that plainly
  names where it came from. The consequence is deliberate and visible: a card
  from this feed opens a match page with no table row and no forecast, and that
  page already says so — the shipped model is fitted on finished matches and
  has no history for a fixture that has not been played.
- **Not every competition.** Nine of the thirty-nine in `configs/leagues.yaml`,
  which is what the free tier serves. A followed competition outside them is
  dropped from the filter — there is no code here to send for it — and a reader
  following only such competitions is told so in a sentence.
- **Not on the feed's clock.** The feed indexes by UTC; the dashboard asks
  about the host's today. On a machine at UTC+05:30 those are different days
  for five and a half hours out of every twenty-four, so a live centre asking
  the feed for the *local* date goes blank exactly during Saturday evening in
  Europe. Fixtures therefore carry host-local date and kick-off (`20:30 IST`,
  labelled because a server-rendered page shows the server's clock), the live
  window is anchored to UTC and filtered by status rather than by date, and a
  requested window is widened a day at each end and narrowed back in the
  answer.
- **Not one request for any window.** The feed refuses a range wider than ten
  days (`400 Specified period must not exceed 10 days`), so a longer ask is
  split into consecutive requests rather than truncated. The shipped dashboard
  asks for seven days, which is one request. Its `dateTo` is an *instant*, not
  a day — `09-05..09-05` answers nothing — so every date in the module is an
  inclusive day with one conversion at the wire, and chunks that meet at
  midnight are merged by `match_id` rather than showing a match twice.
- **Not a corrected feed.** Statuses are rendered as given, including when they
  lag (a 16:45 kick-off was still `IN_PLAY` at 22:40 on the live API).
  Inferring "that must have finished" from the clock would be the dashboard
  inventing a result.
- **Not a minute on this plan.** No row the free tier returned carried
  `minute`, so a live card reads `live` rather than `63'` — `ui.match_card`'s
  existing fallback, which needed no change.
- **Not a rate-limit problem.** Ten calls a minute against a Streamlit script
  that reruns on every click, so answers are memoised for a minute at module
  level — the context builds a fresh provider on every rerun, so an instance
  cache would be one that is empty every time it is read.
- **Not in-play modelling.** The card carries a minute and a score; the
  probabilities beside it are pre-match. A model fitted on in-play state is a
  modelling milestone with its own ablation, and `docs/MODEL_CARD.md` is
  explicit that nothing here is.

## The pages

### Home
The four questions in the order a reader asks them: what is on now, what is on
next, what just finished, and what the model can price. Each is a `Section`
carrying either fixtures or the reason there are none — because "no matches
tonight" and "no fixture feed" are the same empty list and completely different
sentences.

Results come from the match table and priceable fixtures from the service;
the two forward sections come from the fixture feed, and are empty with a
reason until one is configured.

### Live centre
The same feed given the whole screen, for the weekend with forty matches
running at once. Plus a short note on what connecting a provider changes, and
what it does not: in-play probabilities are a *different model* from the one
this repository measures, not a rendering change.

### Competitions
Every competition in `configs/leagues.yaml`, grouped by country, each with its
own page — and the page exists without anybody writing it, because it is a
function of the registry. That is the same property
`tests/unit/test_registry.py` asserts about ingestion, extended to the
presentation layer.

Standings are deliberately absent, and the reason is not laziness: a league
table needs each competition's own rules for points, tie-breaks, deductions and
play-offs, and thirty-nine competitions do not share them.

### Match
The analytics page a card links to. Fixture, forecast, and — the section that
matters — **what that probability is worth**: how often forecasts stated in the
same band actually happened, in this competition, from the same reliability
tables the model card is generated from. A stated probability with no measured
reliability beside it is the number this project exists to stop people quoting.

Then form and head-to-head from the match table, and named placeholders for
expected goals, odds, injuries and in-play statistics — each saying which
milestone supplies it and why it is not a rendering problem.

### Search
Clubs, competitions and matches from one box. Three sections rather than one
ranked list: a reader typing "Arsenal" wants the club and a reader typing
"Serie A" wants the competition, and a single ranking would put one of them
second for no reason a person could predict.

### Model
The three panels Milestone 12 shipped, unchanged in substance: the scoreboard,
the reliability diagram filtered by competition and fold, and the
per-competition breakdown. Moved onto their own page so the home page can be
about football and this one about the forecaster.

## Favourites

Competitions and clubs, held in `st.session_state` — per browser tab, lasting
as long as it is open. Every page reads them; none of them touches
`st.session_state`, which is the whole of the Milestone 14 seam: an account
store replaces the four accessors in `dashboard/domain/favourites.py` and
nothing else changes.

Following no league means *all* the football, not none of it. That distinction
is made in exactly one place, `favourites.league_filter`.

## Visual design

Dark first, and only. Not a preference: a page somebody opens on a Saturday
evening is a page that should not flash white, and a second palette is a second
set of contrast ratios to check.

- **Streamlit's own theme** (`.streamlit/config.toml`) owns the chrome — the
  sidebar, the widgets, the dataframe grid. `dashboard/theme.py` adds only what
  Streamlit has no setting for: the cards.
- **Home, draw and away are green, slate and amber**, not the red/green pair
  the outcome labels invite — which is the most common way a football
  visualisation becomes unreadable for one viewer in twelve. The three differ
  in lightness, so the probability bar survives greyscale.
- **Crests are generated**, because there are no club badges in this repository
  and no licence to ship any. Initials on a colour derived from the club's
  name — stable between sessions, because a reader scanning forty cards
  navigates by colour before they read a word. `Fixture.home_crest_url` is the
  one field a badge provider starts filling.
- **Cards are anchors, not buttons.** A grid of forty `st.button` widgets is
  forty round trips to the server; forty links are forty links.

## Configuration

| Variable | Default | |
|---|---|---|
| `DASHBOARD_API_URL` | `http://127.0.0.1:8000` | Where forecasts are asked for. Compose sets `http://api:8000` |
| `DASHBOARD_FIXTURE_PROVIDER` | `none` | Which fixture feed supplies today, upcoming and live: `none` or `football-data.org` |
| `FOOTBALL_DATA_API_KEY` | unset | The key for that feed. Environment only — `configs/config.yaml` is committed |
| `DATA_DIR` | `data` | Where the match table and the report tables are read from |
| `DASHBOARD_PORT` | `8501` | Container only; the bind address is a literal `0.0.0.0` |

There is no setting for which model the pages report on. It is
`src.models.ensemble.SHIPPED`, read from the one place that names it — the same
constant the API and the model card use.

## What it deliberately is not

- **Not a place to retrain anything.** No button here starts a fit. The
  pipelines are commands with manifests, and a fit triggered from a web page is
  a fit nobody can trace to the bytes that produced it.
- **Not a bookmaker comparison.** Closing odds are in the match table and are
  kept off every page: they are the benchmark this project measures itself
  against, and showing both invites the comparison to be made without the
  walk-forward folds that make it meaningful. `docs/EVALUATION.md` is where
  that comparison lives.
- **Not authenticated, and not for public hosting.** It reads local files and
  talks to a local service. `docker-compose.yml` is a local reproduction of a
  deployment, not a deployment — see `SECURITY.md`.
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
