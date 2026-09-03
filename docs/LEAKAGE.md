# Leakage

Every column the model layer will see, traced back to the canonical columns it
actually reads — measured, not declared — and the four probes that keep the
trace true.

The single most expensive mistake available in this project is training on
something that did not exist until after the match it is meant to predict. It
does not look like a mistake: the metrics simply come out better than they
should, and the model is useless on a Saturday morning. So none of what follows
is an assurance that the code was read carefully. It is a set of properties
that are tested, on every run of the test suite, over every producer that
exists.

---

## What the model layer sees

Three sources, and only three:

| Source | Columns | Where it comes from |
|---|---|---|
| Canonical, pre-match | `date`, `home_team_id`, `away_team_id`, `competition_id`, `country`, `season`, `tier` | `PRE_MATCH_COLUMNS` in `src/ingestion/base.py`. Knowable before kick-off by definition. |
| Ratings | 10 columns | `src/ratings/`, one row per match, from matches strictly earlier. |
| Features | 20 columns | `src/feature_engineering/`, windows over matches strictly earlier. |

And two blocks that are deliberately withheld:

- **`odds_home`, `odds_draw`, `odds_away`.** Genuinely pre-match, and still not
  a feature. A model given the closing line learns to copy the bookmaker, which
  inflates its apparent skill and collapses the moment odds are unavailable.
  They are the benchmark this project measures itself against, so spending them
  as an input would turn that comparison into a comparison with itself. Enforced
  by `benchmark_leaks()`, which fails the suite if any derived column moves when
  a price is rewritten.
- **`referee`, and every post-match column, used directly.** They exist in the
  canonical table because they are the raw material for *lagged* features. The
  distinction is temporal, not columnar, so it cannot be enforced by leaving
  data out — it is enforced by the probes below.

---

## The four probes

All four live in `src/validation/temporal.py` and are generic over
`Callable[[DataFrame], DataFrame]`. None of them needs to know how a value was
produced, which is the point: a rolling mean with an off-by-one window, a
normalisation over the whole table, and a rating updated before it is read all
look correct and all leak.

**1. Prefix invariance.** Truncate the input after match *n*, recompute, and
every surviving row must be byte-identical. Anything that consulted a later
match moves, and so does any statistic taken over the whole table — a mean over
303,517 rows is a different number when there are 50,000 of them.

**2. Outcome independence.** Rewrite one match's scoreline to a 5–0 in
whichever direction it did *not* go, and rows up to and including that match
must not move. This is what prefix invariance cannot see: a computation that
reads match *n*'s own result while emitting match *n*'s features is perfectly
prefix-invariant, because truncation never removes the row it is cheating with.

Both were verified against planted leaks of both kinds before being trusted,
and each is blind to what the other finds.

**3. Split boundary.** The two above test a derivation. A split is the other
place the same leak lives, and the harder one to see, because nothing about the
output looks wrong. No training row may be dated at or after any evaluation
row, and no match may appear in both halves. Ties fail: a full Saturday
programme is one round, and a model trained on the 3pm results is not entitled
to predict the 5.30 kick-off. Milestone 7 owns the splits; the probe is here,
tested, waiting for them.

**4. Observed reads.** Rewrite one input column, recompute, and whatever moved
read it. This is the measured counterpart to the feature registry's
hand-written `reads` declaration — the one part of that registry that can be
wrong without anything noticing, since a typo there reclassifies a leaking
feature as safe. The suite asserts the declaration *covers* what was measured.

---

## Every producer, not every producer we remembered

Milestones 4 and 5 each run the probes inside their own pipeline, over their
own list of producers. A list is a thing you can forget to add to: a builder
wired into a pipeline but omitted from its probe call would ship unverified,
and nothing in the output would say so.

So `src/validation/leakage.py` does not take a list. It walks `src.ratings` and
`src.feature_engineering` and finds every class satisfying the producer
contract, constructed with the same defaults the pipelines construct them with.
A producer that exists is a producer that gets probed. `tests/unit/
test_leakage_suite.py` runs both probes over each of them on every test run,
which means on every CI run, which means a producer added on a branch is probed
by the same command its author already runs locally.

The mirror check closes the other half: `check_defaults_are_complete()` fails if
a discovered producer is in no pipeline's defaults. That is not a leak, but it
is a column the model layer expects and will not get, and it would otherwise
surface milestones later as a table of nulls.

---

## The measured trace

Over the most recent 1,200 matches of ENG_1 and ESP_1 — 2,400 in all. Every
probe held.

| Column | Producer | Reads (measured) | Post-match inputs |
|---|---|---|---|
| `h2h_matches` | head_to_head | `away_team_id`, `date`, `home_team_id` | — |
| `h2h_home_points` | head_to_head | `away_goals`, `away_team_id`, `date`, `home_goals`, `home_team_id` | `away_goals`, `home_goals` |
| `home_matches_played` | team_history | `away_team_id`, `date`, `home_team_id` | — |
| `away_matches_played` | team_history | `away_team_id`, `date`, `home_team_id` | — |
| `home_form_points_5` | team_history | `away_goals`, `away_team_id`, `date`, `home_goals`, `home_team_id` | `away_goals`, `home_goals` |
| `away_form_points_5` | team_history | `away_goals`, `away_team_id`, `date`, `home_goals`, `home_team_id` | `away_goals`, `home_goals` |
| `home_goals_for_5` | team_history | `away_goals`, `away_team_id`, `date`, `home_goals`, `home_team_id` | `away_goals`, `home_goals` |
| `away_goals_for_5` | team_history | `away_goals`, `away_team_id`, `date`, `home_goals`, `home_team_id` | `away_goals`, `home_goals` |
| `home_goals_against_5` | team_history | `away_goals`, `away_team_id`, `date`, `home_goals`, `home_team_id` | `away_goals`, `home_goals` |
| `away_goals_against_5` | team_history | `away_goals`, `away_team_id`, `date`, `home_goals`, `home_team_id` | `away_goals`, `home_goals` |
| `home_shots_for_5` | team_history | `away_shots`, `away_team_id`, `date`, `home_shots`, `home_team_id` | `away_shots`, `home_shots` |
| `away_shots_for_5` | team_history | `away_shots`, `away_team_id`, `date`, `home_shots`, `home_team_id` | `away_shots`, `home_shots` |
| `home_shots_against_5` | team_history | `away_shots`, `away_team_id`, `date`, `home_shots`, `home_team_id` | `away_shots`, `home_shots` |
| `away_shots_against_5` | team_history | `away_shots`, `away_team_id`, `date`, `home_shots`, `home_team_id` | `away_shots`, `home_shots` |
| `home_venue_points_5` | team_history | `away_goals`, `date`, `home_goals`, `home_team_id` | `away_goals`, `home_goals` |
| `away_venue_points_5` | team_history | `away_goals`, `away_team_id`, `date`, `home_goals` | `away_goals`, `home_goals` |
| `home_rest_days` | team_history | `away_team_id`, `date`, `home_team_id` | — |
| `away_rest_days` | team_history | `away_team_id`, `date`, `home_team_id` | — |
| `home_matches_14d` | team_history | `away_team_id`, `date`, `home_team_id` | — |
| `away_matches_14d` | team_history | `away_team_id`, `date`, `home_team_id` | — |
| `dc_home_lambda` | dixon_coles | `away_goals`, `away_team_id`, `competition_id`, `date`, `home_goals`, `home_team_id` | `away_goals`, `home_goals` |
| `dc_away_lambda` | dixon_coles | `away_goals`, `away_team_id`, `competition_id`, `date`, `home_goals`, `home_team_id` | `away_goals`, `home_goals` |
| `dc_prob_home` | dixon_coles | `away_goals`, `away_team_id`, `competition_id`, `date`, `home_goals`, `home_team_id` | `away_goals`, `home_goals` |
| `dc_prob_draw` | dixon_coles | `away_goals`, `away_team_id`, `competition_id`, `date`, `home_goals`, `home_team_id` | `away_goals`, `home_goals` |
| `dc_prob_away` | dixon_coles | `away_goals`, `away_team_id`, `competition_id`, `date`, `home_goals`, `home_team_id` | `away_goals`, `home_goals` |
| `elo_home` | elo | `away_goals`, `away_team_id`, `home_goals`, `home_team_id`, `season` | `away_goals`, `home_goals` |
| `elo_away` | elo | `away_goals`, `away_team_id`, `home_goals`, `home_team_id`, `season` | `away_goals`, `home_goals` |
| `elo_expected_home` | elo | `away_goals`, `away_team_id`, `home_goals`, `home_team_id`, `season` | `away_goals`, `home_goals` |
| `elo_home_played` | elo | `away_team_id`, `home_team_id` | — |
| `elo_away_played` | elo | `away_team_id`, `home_team_id` | — |

Nothing reads `result`, `ht_*`, `referee`, or an odds column. Every post-match
input in the right-hand column is read through a window that ends strictly
before the row it is filling, which is what probes 1 and 2 establish.

### Four things the trace says that reading the code does not

**Elo reads `season` and not `date`.** Its walk is ordered by the frame, and
the only calendar fact it consults is the season label, for carry-over. So an
Elo rating is invariant to *when* matches were played and sensitive only to the
order — which is correct for an online update, and worth knowing before anyone
tries to add a time-decay term to it.

**Dixon-Coles reads `competition_id` and Elo does not.** Dixon-Coles refits per
competition; Elo keeps one pool per country. It gets that pooling for free from
`home_team_id`, which is already country-scoped so a promoted club keeps one id
across divisions — so Elo never reads `country` at all. The partition is in the
identifier, not in a `groupby`.

**`home_venue_points_5` does not read `away_team_id`, and its away counterpart
does not read `home_team_id`.** Venue form is a team's record at its own end of
the ground, and the opponent is genuinely irrelevant to it. This is the
asymmetry the reshape is most likely to get wrong, and it shows up here as an
absence rather than as a number that has to be checked by eye.

**`elo_*_played` reads no result at all.** It counts appearances, so it cannot
leak whatever it does with them — the same conclusion the feature registry
reaches for `*_matches_played` by declaration, reached here by measurement.

---

## What the trace cannot tell you

The measured reads are a **lower bound**, and the two ways they fall short are
both properties of the sample rather than of the code:

- **A column that is entirely null cannot be perturbed.** Run this audit over
  ENG_1's first 2,000 matches and it reports that no feature reads shots —
  correctly, for that slice, because the provider carried no shot data before
  2000/01. The sample is taken from the most recent matches for exactly this
  reason.
- **A column that does not vary cannot be perturbed either.** A single-
  competition sample cannot show that Dixon-Coles partitions on
  `competition_id`, because rewriting a constant leaves one group either way.
  Hence two competitions, in two countries.

So the assertion the suite makes is that a declaration *covers* what was
observed, never that the two are equal. Over-declaring is safe; under-declaring
is what reclassifies a leak as safe.

There is also a class of leak none of this reaches: one in the canonical table
itself. If the provider published a corrected scoreline after the fact and the
ingest overwrote history with it, every probe here would still pass, because
every probe recomputes from the table it is given. That is the ingest layer's
problem and is handled there, by the manifest.

---

## Re-running it

```bash
python scripts/audit_columns.py --competition ENG_1 --competition ESP_1 --markdown
```

About three and a half minutes on that sample, nearly all of it Dixon-Coles
refitting — the audit recomputes every producer twice per canonical column.
Exits non-zero if a probe fails, if a feature reads more than it declares, or if
anything reads the bookmaker's price. The table above is its output.

The cheap half runs with the suite:

```bash
make test        # includes tests/unit/test_leakage_suite.py
```
