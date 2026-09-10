# Match Outcome Predictor

Calibrated home / draw / away probabilities for 39 professional
football competitions, from ingestion through to a served API and dashboard.

[![CI](https://github.com/Vanshcloud/Match-Outcome-Predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/Vanshcloud/Match-Outcome-Predictor/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.13-blue)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

> **Status: Milestone 20 of 20 — the platform: a live fixture feed, saved favourites, live tracking, the closing line beside every forecast, both registered squads, the served forecasts scored against what happened, and fixtures priced before kick-off so that score has out-of-sample matches in it.**
> **303,517 matches** across 39 competitions, 27 countries and 33 years reduce
> to one canonical schema, queryable through a storage interface and checked by
> **24 validation rules** on every ingest. Two ratings and **twenty features**
> run over it, every one proved causal by recomputation rather than by review,
> and **six model families** are scored on walk-forward folds against the
> closing line. A blend of the three whose errors disagree reaches **1.0156 log
> loss** on the 59,001 matches every forecaster could price, against **0.9993**
> for the bookmaker — 0.0121 of the original 0.0284 gap closed, and the last
> 0.0163 looking more like missing information than missing capacity. Three
> attribution methods now say which feature blocks that model uses, a
> **generated model card** says where its probabilities are honest and where
> they are not, and that model is now **served over HTTP** from a container —
> with the card's limitations reachable from the response, and every prediction
> saying whether the fixture was inside the served model's own training window.
> A **dashboard** is the presentation layer over all of it: results, forecasts,
> competitions and search, with the pooled calibration error the card reports
> as one number filterable to the competitions and years it is an average
> over — and every empty section naming the provider that would fill it.

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
   is measured here with reliability tables, not asserted — Milestone 9
   measures it, and reports that the scalar which fixes it does not improve the
   score.
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
all — two points of spread and not even monotone. The ablation settled it:
worth 0.0003 of log loss, kept for the interaction and nothing else.

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
rating is not strength, and is worth about the same amount everywhere. That was
the target the model zoo took aim at, and the two sections below say how much
of it six families, a blend and a calibration layer actually closed.

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

## Ensembling and calibration

Two layers over the zoo, and the interesting result is how little they are
worth.

**The blend's members are chosen on the correlation of their errors**, not on
their scores. Per-match log loss, correlated pairwise over the tuning slice,
puts the four tree-based families between 0.9934 and 0.9960 of each other —
substitutes, which is the quantitative form of "the top four are within
0.0003" — logistic regression at 0.9857, and the MLP alone at 0.92. The
threshold sits in the empty band between 0.9857 and 0.9934 and admits **XGBoost,
logistic regression and the MLP**: the best model, a mediocre one, and the worst
one in the zoo.

| 59,001 matches | log loss | RPS | calibration error |
|---|---:|---:|---:|
| Bookmaker closing odds | **0.9993** | **0.2031** | — |
| Blend of three, calibrated | **1.0156** | **0.2082** | **0.0015** |
| Blend of three | 1.0156 | 0.2082 | 0.0045 |
| CatBoost, the best single family | 1.0159 | 0.2083 | — |
| XGBoost, calibrated | 1.0162 | 0.2084 | 0.0020 |
| XGBoost | 1.0161 | 0.2084 | 0.0037 |

**The blend beats every family that went into it, and the one that did not** —
by 0.0005 over its best member and 0.0002 over CatBoost. That is the argument
for choosing on error correlation rather than on score, and it is also very
little: per competition the blend beats XGBoost in 28 of 39, not 39.

**Calibration buys reliability, not loss.** One temperature per fold, fitted on
the last year of that fold's *training* half with the model refitted on
everything before it, so the evaluation half is never read. It halves the gap
between what the model states and what happens — 0.0037 to 0.0020 — and moves
log loss by 0.00008, in the wrong direction. That is the right answer rather
than a disappointment: log loss is a proper scoring rule and these models were
fitted on it, so what was left was a small overconfidence the score barely
charges for and a reliability table shows immediately.

Together the two layers close **0.0003** of the 0.0165 Milestone 8 left,
leaving 0.0163 to the closing line. Two milestones of model work have bought
0.0121 of the original 0.0284, and the last two layers bought 2% of that — the
strongest evidence yet that what remains is information this project does not
have rather than modelling it has not done.

## Explainability, and the model card

Three ways of asking what a feature block is worth, and they disagree:

| Block | breaking it | its share of the arithmetic | never having had it |
|---|---:|---:|---:|
| `elo` | **+0.0395** | **38.0%** | +0.0021 |
| `form` | +0.0087 | 37.8% | **+0.0033** |
| `dixon_coles` | +0.0073 | 19.9% | +0.0016 |
| `schedule` | +0.0001 | 2.4% | +0.0003 |
| `head_to_head` | +0.0001 | 2.0% | +0.0001 |

Permutation importance shuffles a block at prediction time; SHAP decomposes the
model's own arithmetic; the ablation retrains without the block. **The first
two rank elo above form and the third ranks form above elo** — which is the
substitution Milestone 8 measured, appearing as a number. The model reaches for
elo hardest and can most easily do without it, because Dixon-Coles says most of
the same thing; nothing substitutes for form. A block whose numbers diverge has
a backup, and a block whose numbers agree is load-bearing.

Where all three agree they agree completely: `schedule` and `head_to_head` sit
at the noise floor by every method, on every family, which settles the question
Milestone 5 left open.

**[docs/EXPLAINABILITY.md](docs/EXPLAINABILITY.md)** has the per-family tables
and the method notes.

**[docs/MODEL_CARD.md](docs/MODEL_CARD.md)** is generated from the runs that
measured the model, the way the dataset card is generated from the data: what
it was trained on, what it scores, where its probabilities are honest, and what
it must not be used for. Two breakdowns Milestone 9 deliberately left pooled
are in it — per class, where the draw column is the *most* reliable and the
least useful (the model states about a quarter every time, and about a quarter
of matches are drawn), and per competition, where the Argentine cup is the
least reliable of the 39 at 0.0270 against a pooled 0.0015.

## Serving it

A FastAPI service over the shipped blend: one fixture in, three calibrated
probabilities out, in a container that runs as a non-root user and mounts its
model rather than baking it in.

Two things about it are worth stating before the endpoint list, because both
are decisions rather than defaults.

**It prices the fixtures in the feature table, and does not compute features on
demand.** Recomputing a design row inside a request handler would be a second
implementation of the feature layer living outside every probe that guards the
first — and the whole Milestone 6 argument is that a leak has no symptom, it
simply makes the model look better. So the service reads the row the audited
pipeline wrote. Milestone 20 put *unplayed* matches in that table rather than
making an exception to the rule: `make fixtures` runs the same builders over
the history with next week's fixtures appended, and the request path is still a
lookup.

**Every response says whether the fixture was in the model's training window.**
`make model` fits the blend on the whole history, because that is the model you
would want to serve; the 1.0156 on this page is measured walk-forward, each
fold trained strictly earlier than what it scores. Those are two different
numbers about two different models, and one boolean on every response is what
keeps them from being quoted as one.

| | | |
|---|---|---|
| `POST` | `/predict`, `/predict/batch` | Fixtures priced, named by id or by competition + clubs + date |
| `GET` | `/fixtures` | Which matches can be priced — there is no other way to find out |
| `GET` | `/health`, `/version` | Readiness per component; the model's provenance and library drift |
| `GET` | `/model-card/limitations` | What it must not be used for, served from the card's own text |
| `GET` | `/metrics` | Prometheus exposition — requests, latency, readiness, cache |

`/health` is **200 while the process is alive** and reports `degraded` when the
artefact or the tables are missing, which is exactly what a clean checkout
produces. The container starts anyway and names the command that fixes it, so
the first thing a new reader tries fails with a sentence rather than a stack
trace. Served predictions are written to PostgreSQL with the inputs they were
made from — the first application state in this project, and the condition
Milestone 3 set for adding a second store — and a log that refuses never fails
a request.

A priced fixture is cached, and the cache rests on one fact rather than a
guess: the artefact and the feature table are both loaded once in the lifespan
and never reloaded, so a match id names one design row that one fitted model
turns into one triple of probabilities for the life of the process. It can only
serve the same answer sooner — 7.51 ms to **1.71 ms** on the real table — and
never a stale one. The timestamp is not cached, because `predicted_at` says
when this service answered.

**[docs/API.md](docs/API.md)** has the request shapes, the status codes, the
caching and what the service deliberately is not.
**[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** has the published images, the
probes, what to scrape and three alerts worth having.

## The dashboard

A football dashboard over a prediction engine it is only ever a client of.
Every measurement of the model is produced by `src/pipelines`, every
probability by the service, every result read from the canonical match table —
and `docs/MODEL_CARD.md` is generated from the same functions, so a number on
the page and the same number in the card are the same number from the same
code.

**Four layers, pointing one way.**

```
views  ──▶  services  ──▶  providers  ──▶  domain
```

`domain` is what a match and a forecast *are*, with no I/O. `providers` is
where football comes from — one protocol per kind of source. `services` is
orchestration and caching. `views` is Streamlit, and is the only layer that
would be rewritten if this became a React client reading the same API. CI
asserts the arrows, including the one that matters most: **no view names a
provider.**

**Three data paths, and the split is the design.** Results come from the match
table, because results are what the provider publishes and the API deliberately
serves no scoreline. Reports are read from disk, because a measurement that
already exists should be read rather than recomputed. Forecasts come over HTTP,
because two processes that both unpickle the artefact are two implementations
of "what does the model say". The page therefore works with the service down
and with no data at all: each section names which of the three is missing and
the command that produces it. CI enforces the boundary — `dashboard` may not
import `api`, and the dashboard image does not contain it.

| Page | |
|---|---|
| Home | Live, upcoming, just finished, and what the model can price — each a section that carries either fixtures or the reason there are none |
| Live centre | The same feed given the whole screen, for a weekend with forty matches at once |
| Competitions | Every league in `configs/leagues.yaml`, each with a page nobody wrote |
| Match | The forecast, **what that probability is worth**, the closing line and what a gap from it measures, expected goals, form, head-to-head |
| Search | Clubs, competitions and fixtures from one box |
| Model | The scoreboard, the filterable reliability diagram, the per-competition breakdown |

**Today's matches, upcoming fixtures and live scores come from a second
feed** — this project ingests *results*, and a match that has not been played
is in no table here. Milestone 13 connected football-data.org behind
`FixtureProvider`, which cost exactly what Milestone 12 said it would: one
class, one entry in `FIXTURE_PROVIDERS`, one environment variable, and no view
moved.

```bash
export FOOTBALL_DATA_API_KEY=...            # free: football-data.org/client/register
export DASHBOARD_FIXTURE_PROVIDER=football-data.org
```

Without a key the same sections render from a real implementation that returns
nothing and states why. **Nothing invents a fixture**: a plausible generated
match would put a game on the screen that is not being played, and a reader who
catches that once stops believing the real rows too. Nor is that feed a second
ingestion source — nothing it returns is written to a table, joined to one or
read by a model, and its ids are prefixed `fdorg-` rather than shaped like
canonical ones they would never match.

**The match page is the one worth opening.** Beside three calibrated
probabilities it reports how often forecasts stated in the same band actually
happened, in that competition, from the same reliability tables the model card
is generated from. A stated probability with no measured reliability beside it
is the number this project exists to stop people quoting.

**The closing line is on that page too, and so is what disagreeing with it is
worth.** Milestone 12 kept the odds off every page on the grounds that showing
both invites the comparison to be made without the walk-forward folds that make
it meaningful. Milestone 17 satisfied that objection rather than overruling it:
the comparison is made *with* the folds, and the page shows the answer.

Over 61,889 out-of-sample forecasts, grouped by how far the model was from the
price:

| Apart | n | Model | Market | Model − market | Model better |
|---|---:|---:|---:|---:|---:|
| <2% | 9,628 | 0.9881 | 0.9872 | **+0.0009** | 49.2% |
| 5–10% | 20,874 | 1.0199 | 1.0059 | +0.0140 | 46.4% |
| >20% | 829 | 1.0810 | 0.8975 | **+0.1835** | 35.5% |

**Where the model agrees with the line it is level with it**, and the 0.0163
project-level deficit lives almost entirely in the matches where it does not.
In the widest band the market gets *sharper* — 0.8975 — while the model gets
worse. So a gap is not an edge; it is the best available estimate of how wrong
this model is about that fixture, and the panel says so in those words. There
is no value detector here, and that is a measurement rather than caution.
[docs/EVALUATION.md](docs/EVALUATION.md) has all five bands.

**Expected goals are the goal model's, and are labelled as such.** Dixon-Coles
fits two Poisson rates per fixture and they were already in the ratings table;
the placeholder that used to sit in that panel claimed this project "fits none",
which was wrong. They are not xG off a shot map — nothing here ingests one —
and the difference is stated on the page rather than blurred.

**Both squads are on it as well, and the panel is careful about which word it
uses.** Milestone 18 was scheduled as *availability, injuries, transfers*; what
a reachable source answers is who is **registered**. football-data.org has no
injury endpoint at any tier, and its free plan answers a finished match with an
empty `lineup` and `bench` — measured, not read off a price list. So the fifth
protocol is called `SquadProvider`, the panel says *registered*, and the
section for team sheets and injuries now reads "no source" rather than naming a
milestone.

It is also the first milestone that had to reconcile two vocabularies. Nothing
is joined by id — Milestone 13's rule stands — but a panel on a match page has
to find this club in that feed, and the ingested table says "Hull" where the
feed says "Hull City AFC". Two rules, exact then unique-subset, over names with
case, punctuation, club suffixes and **accents** folded: **146 of 164 clubs**
across the current season of all nine covered competitions, measured against
the live API. Folding accents alone was worth 14 of them. Every remaining miss
is an abbreviation — "Nott'm Forest" against "Nottingham Forest" — and it is a
sentence saying the lookup missed rather than a fuzzy match putting another
club's squad on the page. Nothing on that panel reaches the model: no table
here has ever held a player's name.

**What the service actually served is scored too, and mostly it reports that
it cannot answer yet.** Milestone 19 reads the prediction log back, joins the
matches that have since been played, and puts the served log loss beside the
walk-forward one for the same model. That comparison is the only definition of
drift here — not a distance between feature distributions, which measures that
an input moved rather than that the forecasts got worse.

The finding is a number rather than a verdict. Per-match log loss has a
**standard deviation of 0.3976** across the 62,036 walk-forward forecasts, so a
mean over a handful of served ones is noise:

| Shift in log loss | Scored forecasts needed to see it |
|---|---:|
| 0.05 | 243 |
| **0.0163** — the gap to the closing line | **2,286** |
| 0.01 | 6,073 |

So the report never says "no drift". It says what shift the archive at its
current size *could* have detected, and reports anything smaller as not
evidence. Driven against the compose stack the honest answer today is starker
still: 25 forecasts logged and **none of them scorable**, because the shipped
artefact is fitted through the end of the match table and every fixture the
service can be asked about is one it trained on. The archive starts scoring
when the service is asked about matches *before* they are played — a property
of how it is driven, not of this code.

**The reliability diagram is the one figure this project draws.** Milestone 10
recorded that matplotlib was left out because every figure it would have drawn
was a five-row table. This is the exception and the reason is specific: the
diagram's claim is a diagonal, `y = x` *is* the hypothesis being tested, and a
reader checks a forecast against it by eye in a way a column of signed gaps
does not support. Marker area is the matches in a bin — the same weighting the
calibration error applies, made visible rather than restated.

One enabling change sits behind it. `make card` computed 62,036 per-match
forecasts, used them, and threw them away; it now writes them to
`data/reports/ensemble/forecasts.parquet` with a manifest. The dashboard reads
that instead of spending five minutes on every page load.

**[docs/DASHBOARD.md](docs/DASHBOARD.md)** has the layers, the pages, the
configuration and what the dashboard deliberately is not.

## Storage and validation

The canonical table is read through a `MatchStore`, never by opening a path.
DuckDB implements it as **views over the Parquet** the ingest pipeline writes —
no load step, no second copy, no server. Filters are pushed into the query, so
a point-in-time read (`until="2015-06-30"`) is the interface's own operation
rather than something every caller reimplements against a materialised frame.
CI enforces the seam: nothing outside `src/storage` reads the table directly.

**Twenty-four checks** run on every ingest, over the frame the pipeline just
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
    fixtures.py       the published fixture list, keyed to match
                      the played row it becomes             [Milestone 20] ✅
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
  explainability/   what a block is worth, two ways        [Milestone 10] ✅
    shapley.py        TreeExplainer, aggregated to feature blocks
    permutation.py    break a block, re-score; covers every family
  models/           splits, baselines, the zoo, the blend [Milestone 7-9] ✅
    splits.py         walk-forward folds, cut on the date, probed
    baselines.py      home-always, class prior, Dixon-Coles, the closing line
    dataset.py        the thirty columns, derived from the two registries
    zoo.py            six families, one wrapper, a fresh fit per fold
    tuning.py         Optuna, on matches earlier than every reported fold
    tracking.py       MLflow to a local SQLite file, failure-tolerant
    ensemble.py       members chosen on error correlation, not on score
    calibration.py    one scalar, fitted on a holdout inside the training half
  evaluation/       how good a forecast is, three ways     [Milestone 7-10] ✅
    market.py         the closing line as a forecast, and the
                      disagreement measurement               [Milestone 17] ✅
    archive.py        what was served, scored — and how much
                      archive a verdict needs               [Milestone 19] ✅
    metrics.py        two proper scoring rules and one improper one
    reliability.py    does a stated probability happen at the rate it states
    model_card.py     the card, rendered from data it is handed
  pipelines/        the orchestration each stage exposes
    tables.py         the three tables, joined once for every command
    report.py         the breakdowns, and the card assembled from them
    serving.py        the artefact: fit once, persist, load, look a fixture up
    fixtures.py       design rows for matches not yet played,
                      by the same builders                 [Milestone 20] ✅
  models/artifact.py  the shipped blend, fitted — frames in, arrays out
  storage/predictions.py  served predictions, in PostgreSQL   [Milestone 11] ✅
                      read back and scored by `make archive` [Milestone 19] ✅
api/                the inference service                 [Milestone 11] ✅
  main.py             lifespan, middleware, error mapping, cache policy
  routes.py           seven endpoints, each a call and a return
  service.py          the model, the index, the log and the cache, held once
  metrics.py          the counters, and the exposition   [Milestone 16] ✅
  schemas.py          the request and response models, and the OpenAPI document
dashboard/          the presentation layer               [Milestone 12] ✅
  app.py              the shell: theme, sidebar, six declared pages
  domain/             what a match, a forecast and a competition are
    identity.py         who the reader is: a profile, or an account
    store.py            what they follow, kept between visits  [Milestone 14] ✅
  providers/          where football comes from — one protocol per source
    historical.py       results, and the closing line  [Milestone 17] ✅
    api.py              forecasts, from the service over HTTP
    football_data_org.py  fixtures, live scores        [Milestone 13] ✅
                          and registered squads        [Milestone 18] ✅
    webhook.py          where a goal is posted             [Milestone 15] ✅
    null.py             no feed and no transport: nothing, and why
  services/           orchestration and caching over the providers
    watch.py            what changed since this tab last looked [M15] ✅
    market.py           the price, the goal rates, the verdict [M17] ✅
  views/              Streamlit, thin and swappable
  ui.py               cards, probability bars, crests, form strings
  charts.py           the reliability diagram, and two honest conveniences
  client.py           the service, over HTTP — `api` is never imported
```

`api` imports `src`; nothing in `src` imports `api`. CI enforces that, and the
tighter rule that keeps the service off the feature layer: a design row is
thirty columns the leakage suite probes on every build, and rebuilding one
inside a request handler would put an unprobed copy of that layer on the
request path. **[docs/API.md](docs/API.md)** has the endpoints and the
reasoning.

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
make train     # fit the six model families over the folds (~5 minutes)
make ablation  # what each feature block is worth (~10 minutes)
make ensemble  # the blend, the calibration scalar, the reliability tables (~20 min)
make explain   # what each feature block is worth, by SHAP and permutation (~2 min)
make card      # regenerate docs/MODEL_CARD.md (~5 minutes)
make model     # fit the shipped model on the whole history and persist it (~1 min)
make api       # serve it at http://127.0.0.1:8000/docs
make dashboard # the reports and a live price at http://127.0.0.1:8501
make docker-run # the API and its prediction log, via compose
make test      # unit tests — no network, no data needed
make test-int  # integration tests — needs `make data`
make fixtures  # build design rows for what is about to be played (~12 min)
make price     # ask the running service about them, so the log fills
make archive   # score the served forecasts; needs PREDICTION_LOG_DSN
make invariants # CI's architectural boundary checks, here rather than there
make quality   # ruff + black + mypy + the invariants
```

Requires Python 3.13. From Milestone 8, macOS also needs `brew install libomp`
for LightGBM and XGBoost.

Configuration is `configs/config.yaml`, overlaid by an explicit set of
environment variables (see `.env.example`). Unknown keys are an error, not a
shrug — a misspelt setting fails at load naming the key, rather than appearing
to work forever.

### Reproducing every number in this README

```bash
make reproduce   # the whole project, clean checkout to a served model (~60 min)
```

One command, because nine commands listed in a paragraph is nine chances to
get the order wrong. It runs the stages in the only order they work in, each
reading the table the one before it wrote:

| # | Stage | Command it runs | ~Runtime | What it writes |
|---|---|---|---:|---|
| 1 | Setup | `make setup` | 2 min | `.venv/`, git hooks |
| 2 | Data | `make data` | 15 min | `data/raw/`, `data/processed/matches.parquet` |
| 3 | Ratings | `make ratings` | 10 min | `data/processed/ratings.parquet` |
| 4 | Features | `make features` | 10 sec | `data/features/features.parquet` |
| 5 | Training | `make train` | 5 min | `data/reports/zoo/`, MLflow runs in `models/` |
| 6 | Ensemble | `make ensemble` | 20 min | `data/reports/ensemble/` |
| 7 | Explainability | `make explain` | 2 min | the attribution tables, as markdown on stdout |
| 8 | Model card | `make card` | 5 min | `docs/MODEL_CARD.md` |
| 9 | Servable model | `make model` | 1 min | `models/servable.joblib` and its manifest |

`make validate`, `make audit`, `make backtest` and `make ablation` are
deliberately not in the sequence: none of the nine stages reads what they
write, and each roughly doubles the wall clock. Run `make ablation` before
stage 7 and the attribution tables gain an ablation column; leave it out and
they print without one.

About an hour end to end on a laptop, most of it in stages 2, 3 and 6. The
first is a download and is bounded by the provider rather than by the machine;
the other two walk the folds more than once. Individual stages are re-runnable
on their own — everything is idempotent, and a stage whose inputs have not
changed rewrites the same bytes.

**Dependencies are pinned twice over.** `requirements.txt` and
`requirements-lint.txt` pin the direct dependencies exactly, and `make setup`
installs from them; **`uv.lock` is committed** and pins the transitive closure
with hashes, so `uv sync --frozen` reconstructs the runtime environment the
benchmark numbers on this page were measured in rather than a today's-resolver
approximation of it. Without a lockfile a transitive release moves a fourth
decimal place and the table above quietly stops describing what you would get.
Regenerate it with `uv lock` in the same commit as any change to a
requirements file.

## Engineering standards

| Gate | Tool | Enforced by |
|---|---|---|
| Lint | ruff | `make lint`, CI |
| Format | black | `make format-check`, CI |
| Types | mypy, strict | `make typecheck`, CI |
| Architecture | 18 greps holding the dependency graph in the shape this document claims | `make invariants`, CI |
| Tests | pytest, 100% coverage of `src`, `api` and `dashboard` | `make test-cov`, CI |
| Deprecations | `-W error::DeprecationWarning` | pytest config |
| Authorship | `scripts/hooks/commit-msg` | git hook + CI |
| Serving image | `docker build`, then a live `/health`, `/metrics` and cache directive against it | CI |
| Dashboard image | `docker build`, then a live `/_stcore/health` against it | CI |
| Published image | Both started and asserted against before the push; a tag that disagrees with `src/__init__.py` is refused | Release |

Lint tooling is pinned **exactly**. Unpinned, a formatter release turns CI red
with no code change and disagrees with every local run.

`requirements.txt` grows one milestone at a time. A pinned dependency nothing
imports is a supply-chain surface with no upside.

## Roadmap

| # | Milestone | Status |
|---|---|---|
| 1 | Foundation — config, logging, paths, HTTP, gates | ✅ |
| 2 | Ingestion — adapters, canonical schema, league registry | ✅ |
| 3 | Storage and validation — DuckDB over Parquet, 24 checks, dataset card | ✅ |
| 4 | Ratings — Elo, Dixon-Coles, causality probes | ✅ |
| 5 | Feature engineering — registry, causal windows, 20 features | ✅ |
| 6 | Leakage suite — every producer found and probed, every column traced | ✅ |
| 7 | Splits and baselines — walk-forward CV, RPS/log loss | ✅ |
| 8 | Model zoo — LR, RF, XGBoost, LightGBM, CatBoost, MLP, Optuna, MLflow | ✅ |
| 9 | Ensembling and calibration — error-correlation selection, temperature scaling, reliability | ✅ |
| 10 | Evaluation and explainability — SHAP, permutation importance, model card | ✅ |
| 11 | API — FastAPI, Docker, PostgreSQL for served predictions | ✅ |
| 12 | Dashboard — the presentation layer, four layers deep, six pages | ✅ |
| 13 | Live fixture ingestion — a `FixtureProvider` against football-data.org or equivalent | ✅ |
| 14 | Accounts — authentication, saved favourites and preferences | ✅ |
| 15 | Real-time — in-play updates, notifications, favourite-team alerts | ✅ |
| 16 | Deployment and operations — published images, Prometheus metrics, prediction cache | ✅ |
| 17 | Odds and expected goals — the closing line on the match page, and what a gap from it measures | ✅ |
| 18 | Availability — registered squads on the match page, and the two of those three words no reachable source answers | ✅ |
| 19 | Prediction archive — served forecasts scored against what happened, and how much archive a drift figure would need | ✅ |
| 20 | The loop closed — fixtures priced before kick-off, so the archive has out-of-sample forecasts to score | ✅ |

Research models — TabNet, FT-Transformer, AutoML, benchmarked against the best
GBDT with a written verdict — is unscheduled rather than dropped: it changes
every reported number in this repository and belongs to a milestone with its
own ablation, not to a platform release.

## Security

No personal data, no credentials, no user accounts; the one secret is the
prediction log's DSN, and it is environment-only for that reason. See
[SECURITY.md](SECURITY.md) for what is in scope and how to report something.

## Licence

MIT. See [LICENSE](LICENSE).

Match data is sourced from football-data.co.uk and remains subject to its
terms. No data is redistributed by this repository.
