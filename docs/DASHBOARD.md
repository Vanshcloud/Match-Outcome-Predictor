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

The third path checks *which* service answered. A `/health` returning
`{"status": "ok"}` is not evidence that the thing on the port is this project's
API — on a machine running more than one service, it is quite likely not — so
the body must also carry the component list `api/schemas.py` makes required.

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
running at once, repainting itself every sixty seconds. Plus a short note on
what this page is and is not: in-play probabilities are a *different model*
from the one this repository measures, not a rendering change.

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

Then, since Milestone 17, **what the market said** and **what the goal model
expects** — see [The market, and what a gap from it means](#the-market-and-what-a-gap-from-it-means)
below — and since Milestone 18, **who is registered**: both squads, from the
same feed the live scores come from. Then form and head-to-head from the match
table, and named placeholders for team sheets, injuries, in-play statistics and
shot-quality xG — each saying what it needs and why it is not a rendering
problem.

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

Milestone 19 adds a fourth, **What we served**, and it is the only panel on this
dashboard whose subject is not the backtest: the forecasts the service actually
answered, scored against the results that arrived afterwards. It renders the
size of the archive before the drift figure and never the other way round — a
mean log loss over eleven matches drawn from a distribution with a per-match
spread of 0.4 is noise, and a page that led with it would be read headline
first. A difference under the noise floor is captioned *not evidence of drift*,
with the table saying how many more forecasts it would take. See
[docs/DEPLOYMENT.md](DEPLOYMENT.md#drift-and-how-much-archive-it-takes).

## The market, and what a gap from it means

Milestone 17. Two panels on the match page, and one of them is mostly a
sentence.

### What the market said

The closing line, with the overround removed, beside the model's three
probabilities and under the same probability bar. The decimal price sits under
each percentage and the size of the margin is stated — the book on a typical
match here pays out on about 108% of the stake, and removing that
proportionally is the *transparent* way rather than the most accurate one,
because the favourite carries more of the margin than an equal share.

The de-vig is `src/evaluation/market.py::implied_probabilities`, which is the
same function the backtest's `bookmaker` benchmark calls. One implementation,
so the percentage a reader sees is the percentage the model was scored against.

Odds cover about **81%** of the match table — effectively everything from 2003
and nothing before it — and nothing that has not been played, because this
project ingests results. A fixture without a price says so.

### How far apart, and what that is worth

This is the panel Milestone 12 said should not exist, and the reason it now
does is that the objection was satisfied rather than overruled. Milestone 12's
argument was that *"showing both invites the comparison to be made without the
walk-forward folds that make it meaningful."* So the comparison is made **with**
the folds, in `src/pipelines/report.py::market_comparison`, and the panel shows
the answer instead of the invitation.

The answer is not the one the layout implies. Over 61,889 out-of-sample
forecasts, the model's deficit against the closing line **grows** with the size
of the disagreement:

| Apart | n | Model | Market | Model − market | Model better |
|---|---|---|---|---|---|
| <2% | 9,628 | 0.9881 | 0.9872 | **+0.0009** | 49.2% |
| 2–5% | 21,139 | 1.0136 | 1.0099 | +0.0037 | 48.4% |
| 5–10% | 20,874 | 1.0199 | 1.0059 | +0.0140 | 46.4% |
| 10–20% | 9,419 | 1.0402 | 0.9864 | +0.0538 | 41.9% |
| >20% | 829 | 1.0810 | 0.8975 | **+0.1835** | 35.5% |

Two readings, and the second is the one that matters.

**Where the model agrees with the closing line, it is level with it.** +0.0009
over 9,628 matches is not a deficit anybody would act on. The project-level gap
of 0.0163 that [EVALUATION.md](EVALUATION.md) reports does not come from the
model being uniformly worse; it comes almost entirely from the matches where
the model disagrees.

**A gap is not an edge.** In the widest band the market's own log loss *falls*
to 0.8975 — those are matches it prices confidently and correctly — while the
model's rises to 1.0810. So the honest reading of a wide gap on a fixture is
that this model is more likely to be wrong about that match, and the panel says
so in those words. Nothing on this page suggests a bet, and that is a
measurement rather than a disclaimer.

The five bands are read from `market.parquet`, written by `make card`. Before
that command has run the panel still shows both forecasts and how far apart
they are — it just declines to say what the distance has been worth.

### Expected goals, and the word that is not used

`dc_home_lambda` and `dc_away_lambda` from the ratings table: the two Poisson
rates Milestone 4's Dixon-Coles model fits, with the total and the supremacy
beside them.

**These are not xG.** Nothing in this project has ever seen a shot map — the
ingested feed carries shots and shots on target and no expected-goals column —
so a panel labelled "xG" would be attributing a rival provider's measurement to
a model that made an estimate. The placeholder that used to sit here said this
project "fits none", which was wrong: Dixon-Coles is a goal model, and these
are its rates.

They are worth showing because they are *how the rating thinks*. The
three-class probability this project reports is a sum over a Poisson grid built
from exactly these two numbers, so a reader asking why a forecast leans one way
is looking at its inputs. A fixture before the model's first fit for its
competition has no rates, and says that rather than showing zeros.

## Squads, and the three words that are not interchangeable

Milestone 18. The roadmap called it *player availability, injuries, transfers*.
What shipped is a squad panel, and the gap between those two sentences is the
milestone.

| `DASHBOARD_SQUAD_PROVIDER` | Class | What it answers |
|---|---|---|
| `none` (default) | `providers/null.py` | Nothing, and the reason. |
| `football-data.org` | `providers/football_data_org.py` | `GET /v4/competitions/{code}/teams`: every club in a competition with its registered squad — name, position, date of birth, nationality. Nine competitions, the same `FOOTBALL_DATA_API_KEY`. |

```bash
export FOOTBALL_DATA_API_KEY=...
export DASHBOARD_SQUAD_PROVIDER=football-data.org
```

Its own variable rather than riding on `DASHBOARD_FIXTURE_PROVIDER`, even
though the shipped implementation of both is one feed and one key: a reader who
wants live scores and no squad panel — or the reverse — sets one and not the
other.

### Registered is not available, and available is not selected

- **Registered** is who the club has on its list. This is what ships.
- **Available** is who is fit and not suspended. **football-data.org has no
  injury endpoint at any tier.** There is no URL to be refused.
- **Selected** is the eleven. The free plan answers `/v4/matches/{id}` with an
  **empty** `lineup` and `bench`, on a *finished* match — measured against the
  live API, not inferred from the price list.

So the protocol is `SquadProvider` and not `AvailabilityProvider`: a protocol
named after the question rather than after the answer would be three methods
returning `None`. The panel says *registered*, the caption says what that does
not include, and the section for the other two now reads "no source" instead of
naming a milestone that has been spent.

### The join Milestone 13 said it would not make

Milestone 13 was explicit that its feed is **not joinable to the match table**,
and that has not changed: ids are still prefixed `fdorg-` and nothing this feed
returns is written anywhere. But a squad panel on a match page has to find
*this* club in *that* feed, and Milestone 18 is the first milestone that needs
the two vocabularies reconciled at all.

It is done by name, in `providers/football_data_org.py::pick`, with two rules
and no third:

1. Exact match on the normalised name, short name or three-letter code.
2. Every word of this project's name appearing in the feed's — "Hull" in "Hull
   City", "Forest" in "Nottingham Forest" — and **only when exactly one club
   matches**.

Normalising folds case, punctuation, club-form suffixes (`FC`, `AFC`) and
**accents**: this feed writes "Grêmio", "São Paulo" and "1. FC Köln" where the
ingested table writes them plain. Two spellings of one letter is an encoding
convention rather than a different name, and folding them was worth 14 clubs.

**Measured against the live feed and the real table**, over the current season
of all nine covered competitions — 164 clubs:

| | Clubs matched |
|---|---|
| Exact and subset rules, accents kept | 136 / 164 (82.9%) |
| **Shipped — accents folded** | **146 / 164 (89.0%)** |

| Competition | | Missed |
|---|---|---|
| Premier League | 19/20 | Nott'm Forest |
| Championship | 23/24 | Wolves |
| Bundesliga | 8/9 | Bayern Munich |
| Serie A | 19/19 | — |
| La Liga | 16/19 | Ath Bilbao, Ath Madrid, Espanol |
| Ligue 1 | 16/17 | Rennes |
| Eredivisie | 14/18 | AZ Alkmaar, For Sittard, Nijmegen, PSV Eindhoven |
| Primeira Liga | 15/18 | Guimaraes, Sp Braga, Sp Lisbon |
| Brasileirão | 16/20 | Athletico-PR, Atletico-MG, Botafogo RJ, Flamengo RJ |

Every remaining miss is an **abbreviation**, not an encoding: "Nott'm Forest"
against "Nottingham Forest", "Wolves" against "Wolverhampton", "Sp Lisbon"
against "Sporting CP". No fuzzy distance and no alias table — an alias table is
eighteen hand-written lines that go stale every August, and a fuzzy match is a
coin toss that puts another club's squad on the page. **A miss is a sentence
saying the lookup missed**, which is the same rule as everywhere else here: two
candidates also answer nothing, for the same reason.

### What it costs the feed

One request per competition, not one per club: the competition endpoint answers
with every club *and* every squad, so a match page costs one call and the next
fixture in that competition costs none. Reused for **an hour** against the
fixture feed's minute — a squad moves on a transfer deadline and a score moves
on a goal, and the same ten requests a minute pay for both.

### What it did not touch

No view moved for the odds at Milestone 17 and none moved for this: the panel
is one function on the match page, and `src/` is **unchanged**. Nothing here
reaches the model — no table in this project has ever held a player's name, and
the forecast above the panel was produced by a model fitted on scorelines. A
squad panel that quietly became a feature would be a model change wearing a
presentation change's clothes.

## Live tracking and alerts

Milestone 15. The live strip **repaints itself** — `@st.fragment(run_every=60)`
— and says what changed since it last looked.

A fragment rather than a whole-page rerun: everything else on the home page is
a file read or an HTTP call (the results table, the reliability tables, the
service's fixture list), and repainting all of it every minute to move one
score would be the most expensive way to show the cheapest change. The interval
is matched to the feed's own sixty-second memo rather than chosen, so a refresh
that finds nothing new costs no request at all — which matters on a free tier
of ten a minute.

### Three events, and the rules that keep them honest

| | When |
|---|---|
| 🟢 Kick-off | A match appears in the live answer that was not there before |
| ⚽ Goal | A tracked match's score changed |
| 🔔 Full time | A tracked match has gone from the live answer |

Two of those rules are wrong in the obvious implementation and are worth
stating:

- **The first look announces nothing.** With no previous snapshot there is no
  "since", and a page that toasted a kick-off for a match already an hour old
  would be telling a reader something untrue.
- **An empty answer is not full time.** The feed returns nothing both when
  nothing is in play and when it could not be reached, so a diff that read
  absence as "the match ended" would announce eight final whistles because of
  one rate limit. The feed is asked whether it thinks it answered, and a failed
  look produces no events at all.

A changing *minute* is deliberately not an event. A notification per minute of
a match is a notification a person turns off.

The snapshot lives in `st.session_state` — one per browser tab, because "since
*I* last looked" is a per-tab question, and a snapshot in the profile store
would mean the first tab to refresh silently consumed the second one's news.

### Where an event goes

| `DASHBOARD_NOTIFIER` | Class | What it does |
|---|---|---|
| `none` (default) | `providers/null.py` | Nothing, and says what configuring a transport would add |
| `webhook` | `providers/webhook.py` | One POST per event to `DASHBOARD_WEBHOOK_URL` |

```bash
export DASHBOARD_NOTIFIER=webhook
export DASHBOARD_WEBHOOK_URL=https://hooks.slack.com/services/...
```

The toast is in the page and needs no transport; the webhook is the half that
reaches a phone. One body serves the common receivers — `text` for Slack,
`content` for Discord, and the event's own fields for anything programmatic —
so there is no "which flavour of webhook" setting, which is a setting nobody
can answer without opening the receiving service's documentation anyway.

A transport that refuses is a `False` and a caption, never an exception: the
strip it hangs off is about football, and a webhook that 404s must not cost a
reader the scores. Nothing is retried — `HttpClient` retries only GET and HEAD,
and a POST that failed may already have been acted on, so a duplicate goal
alert can never be this dashboard's doing.

**Alerts exist while something is watching.** The page has to be open. A
process that polls with every browser closed is a different thing with its own
lifecycle — a second consumer of a ten-request budget, reading favourites
outside Streamlit — and it is not in this milestone.

## Favourites

Competitions and clubs, and since Milestone 14 they **outlive the browser
tab**. They are saved against whoever the reader is, in a small JSON file that
`dashboard/domain/store.py` owns.

Who the reader is, is decided in one module — `dashboard/domain/identity.py` —
and there are two answers:

| | Key | When |
|---|---|---|
| **A profile** | `profile:<name>` | Always available. Picked from the sidebar; `Guest` until someone picks otherwise, and `Guest` is a real profile with a real row, so favourites persist for a reader who never opens the picker. |
| **An account** | `account:<verified email>` | Only where the deployment configures an OIDC provider for Streamlit's own `st.login()`. |

The keys are namespaced on purpose. Without the prefixes, someone who typed a
colleague's email address as their profile name would be handed that
colleague's favourites — small here, and exactly the shape of a serious thing
in an application that stored more.

**A profile is not an account, and the page says so.** There is no password: on
a deployment with no provider configured, anyone who can open the page can pick
any profile. That is the right amount of security for something this document
already describes as not for public hosting, and the wrong amount for anything
else — which is what the account half is for.

### Turning accounts on

Two things, neither of which this repository ships:

```toml
# .streamlit/secrets.toml — gitignored, and every value in it is a credential
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "..."
[auth.google]
client_id = "..."
client_secret = "..."
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```
```bash
pip install "streamlit[auth]"      # authlib; requirements-dashboard.txt omits it
```

The sidebar offers a sign-in button only when **both** are present. Streamlit
adds `is_logged_in` to `st.user` only when an `[auth]` section exists, and
`st.login()` raises without authlib — so a button offered on either half alone
is a button that always errors, and the page shows the profile picker instead.

### Where the file goes

`$DASHBOARD_PROFILE_STORE`, or `<DATA_DIR>/dashboard/profiles.json` by default.
The compose file mounts `data/` **read-only** — a container that cannot corrupt
its own inputs is worth more than one that can save a preference to them — so
the image gets a named volume at `/app/profiles` and the variable points there.

A write that cannot land is a sentence in the sidebar, not an exception: a
read-only filesystem is a state a real deployment reaches, and losing a
preference must not lose the page. Reads are equally forgiving — a truncated or
hand-edited file gives a reader with no favourites rather than a stack trace on
every page.

It is a file rather than the PostgreSQL the compose file already runs, and that
is a decision with a ceiling written next to it: that database is the
*service's*, holding served predictions so calibration can be measured, and
reaching it from here would mean `psycopg` in an image that documents its
absence, a connection pool nobody tracks across Streamlit reruns, and a schema
migration for a preference. One JSON file, rewritten whole, last writer wins.
Two people editing different profiles in the same second lose one edit. The
upgrade is that database, and the seam for it is `store.py`.

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
| `DASHBOARD_API_URL` | `http://127.0.0.1:8000` | Where forecasts are asked for. Compose sets `http://api:8000`. The dashboard checks that what answers there is *this* project's API — a `/health` that says `ok` but carries no component list is reported as the misconfiguration it is |
| `DASHBOARD_FIXTURE_PROVIDER` | `none` | Which fixture feed supplies today, upcoming and live: `none` or `football-data.org` |
| `FOOTBALL_DATA_API_KEY` | unset | The key for that feed. Environment only — `configs/config.yaml` is committed |
| `DASHBOARD_SQUAD_PROVIDER` | `none` | Which source supplies registered squads: `none` or `football-data.org`. Its own variable, so live scores and the squad panel are turned on separately |
| `DASHBOARD_PROFILE_STORE` | `<DATA_DIR>/dashboard/profiles.json` | Where saved favourites live. Compose points it at a writable volume |
| `DASHBOARD_NOTIFIER` | `none` | Where match events are sent: `none` or `webhook` |
| `DASHBOARD_WEBHOOK_URL` | unset | The URL `webhook` posts to. A credential — environment only |
| `DATA_DIR` | `data` | Where the match table and the report tables are read from |
| `DASHBOARD_PORT` | `8501` | Container only; the bind address is a literal `0.0.0.0` |

There is no setting for which model the pages report on. It is
`src.models.ensemble.SHIPPED`, read from the one place that names it — the same
constant the API and the model card use.

## What it deliberately is not

- **Not a place to retrain anything.** No button here starts a fit. The
  pipelines are commands with manifests, and a fit triggered from a web page is
  a fit nobody can trace to the bytes that produced it.
- **Not a tipster.** Milestone 17 put the closing line on the match page, and
  the panel beside it says what a gap from that line has been worth: nothing
  good for the model. No page here flags value, suggests a stake or computes an
  expected return, and the reason is a measurement rather than caution — the
  model's deficit against the price grows with the size of the disagreement, so
  the difference a value detector would trade on is the best available
  estimate of this model's own error.
- **Not authenticated by default, and not for public hosting.** It reads local
  files and talks to a local service, and out of the box its profiles are
  names without passwords. Milestone 14 added optional OIDC login for
  deployments that configure one, which changes who favourites belong to — not
  what the pages will show a reader who is not signed in. Every page remains
  readable by anyone who can reach it. `docker-compose.yml` is a local
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
