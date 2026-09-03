# Features

Twenty columns, one row per match, every one of them a summary of what happened
*before* it. What each is, what it is worth, and how the "before" is enforced.

---

## How leakage is prevented

Not by care. By three mechanisms that do not rely on anyone reading the code
correctly.

**1. The window cuts on the date, not on the row.** The obvious implementation
is `shift(1).rolling(k)`, and it is subtly wrong: `shift` counts rows, so two
matches on the same date let the earlier one — earlier only by an arbitrary
tiebreak in the sort — inform the later. Every window here instead ends at the
last row *strictly earlier by date*. Same-day matches cannot see each other,
whatever order they were sorted in.

That is not a hypothetical. The mislabelled-division bug in
[DATA_SOURCES.md](DATA_SOURCES.md) was found by asking whether a team ever
plays twice on one date, and the answer at the time was 2,444 times.

**2. The registry says which features have anything to prove.** Each feature
declares the canonical columns it reads. Whether it *can* leak follows from
that — a feature touching `POST_MATCH_COLUMNS` can, one reading only the
fixture list cannot — rather than from a separate field somebody might set
wrongly. Seven of the twenty read nothing but who is playing and when.

**3. The probes recompute it.** `src/validation/temporal.py` truncates the
input and checks the surviving rows do not move, then rewrites one scoreline
and checks that match's own row does not move. Both run on every build against
a sample competition, and the build exits non-zero if either fails. A
deliberately leaky builder in the test suite proves the failure path works.

**And a fourth, on the table itself.** Every window feature has a companion
count saying how much history it had. A value where the count is zero came from
somewhere it should not have, and the check suite says so over every row that
was written — not just the sampled competition the probes see.

---

## The features

| Group | Feature | Reads | Window |
|---|---|---|---|
| form | `{side}_matches_played` | fixture list | all history |
| form | `{side}_form_points_5` | results | last 5 |
| form | `{side}_goals_for_5`, `{side}_goals_against_5` | results | last 5 |
| form | `{side}_shots_for_5`, `{side}_shots_against_5` | shots | last 5 |
| form | `{side}_venue_points_5` | results | last 5 at that venue |
| schedule | `{side}_rest_days` | fixture list | previous match |
| schedule | `{side}_matches_14d` | fixture list | previous fortnight |
| head to head | `h2h_matches` | fixture list | all meetings |
| head to head | `h2h_home_points` | results | last 5 meetings |

**Form spans competitions.** A club's last five are its last five, league or
cup. A per-competition window restarts a promoted team at zero and pretends the
midweek tie never happened, which is not what a form table means.

**Coverage**, measured over the 303,517-match ingest:

| Feature | Coverage | Why not 100% |
|---|---:|---|
| `home_matches_played`, `home_matches_14d`, `h2h_matches` | 100% | A count is knowable even when it is zero |
| `home_form_points_5` | 99.8% | Every team's first ever match |
| `home_venue_points_5` | 99.6% | And its first at that venue |
| `home_rest_days` | 99.8% | Nothing to measure rest from |
| `h2h_home_points` | 91.5% | Two clubs that have never met |
| `home_shots_for_5` | 42.8% | The provider added shot data around 2019/20, and the secondary feed has none at all |

---

## What each is worth

### Venue form: worth having, and exactly symmetric

| | Overall form | At that venue | Difference |
|---|---:|---:|---:|
| Home side | 1.343 | **1.618** | +0.275 |
| Away side | 1.392 | **1.119** | −0.273 |

A home team takes a quarter of a point per game more at home than it does on
average, and an away team a quarter less away. That the two are the same size
in opposite directions is the check that the reshape is not confusing the two
sides — an error that would leave both numbers plausible on their own.

Elo carries a single global home-advantage constant. This is the part it cannot
express: which particular clubs are transformed by their own ground.

### Form gap: strongly monotone, and draws behave correctly

Home win rate by `home_form_points_5 − away_form_points_5`:

| Form gap | Matches | Home | Draw | Away |
|---|---:|---:|---:|---:|
| below −1.2 | 38,186 | 31.7% | 27.2% | **41.1%** |
| −1.2 to −0.6 | 51,256 | 38.4% | 27.7% | 33.9% |
| −0.6 to −0.2 | 37,911 | 41.9% | 27.8% | 30.2% |
| −0.2 to 0.2 | 61,169 | 45.1% | **27.5%** | 27.3% |
| 0.2 to 0.6 | 38,959 | 48.7% | 26.8% | 24.5% |
| 0.6 to 1.2 | 48,831 | 52.7% | 25.6% | 21.7% |
| above 1.2 | 26,201 | **61.1%** | 22.2% | 16.7% |

Home win rate moves from 31.7% to 61.1% — a thirty-point spread from one
feature. And the draw rate is highest between evenly matched sides and falls at
both extremes, which is what football says should happen and is not something
the feature was built to produce.

### Rest days: no marginal signal at all

The honest one.

| Home team's rest advantage | Matches | Home win |
|---|---:|---:|
| Much less (4+ days fewer) | 9,423 | 44.9% |
| Less | 16,640 | 45.6% |
| Equal | 249,981 | 45.0% |
| More | 19,050 | **43.7%** |
| Much more (4+ days more) | 6,970 | 44.7% |

Two points of spread, and not even monotone — the side with *more* rest wins
slightly less often. Whatever fixture congestion does to a team, it is not
visible in the marginal.

It is kept, for two reasons and neither of them is hope. A flat marginal is not
a flat *interaction*: a tired team facing a strong one is a different case from
a tired team facing a weak one, and a tree model is exactly the thing that can
find that. And the column costs nothing to carry. Milestone 8's ablation is
where it earns its place or is dropped, with a real model to measure against
rather than a cross-tab.

---

## What is deliberately not here

Nothing derived from ratings. Elo and Dixon-Coles live in their own table and
join on `match_id`; duplicating them here would mean two copies of a number
rebuilt on different cadences.

Nothing derived from odds. They are pre-match and therefore safe, and they are
still reserved as the benchmark — see [RATINGS.md](RATINGS.md).

No weather, injuries, lineups, travel or attendance. None is available at match
level across these competitions from any free source, and the registry is where
one would go the day it becomes available.
