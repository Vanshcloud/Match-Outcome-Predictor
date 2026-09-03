# Ratings

What each rating is, what it measured, and what was tried and thrown away.

Every number here comes from walking all **305,499 matches** in date order, with
each prediction made from prior matches only. That makes them out-of-sample
figures over the whole table rather than fits, which is the only kind worth
printing.

---

## Elo

One pool **per country**, not per competition. Team identity is already
country-scoped so a promoted club keeps one id across divisions, and resetting
its rating at promotion would throw away the history that makes the rating
worth anything.

The honest cost: league fixtures never cross divisions, so a country's
divisions are connected only through promotion, relegation and the occasional
cup tie. The absolute level of a lower division's ratings drifts against the
top flight's more than a fully connected pool's would. Ratings stay comparable
*within* a division, which is where every fixture is played.

### What each refinement is worth

Mean squared error of the expected score against the actual 1 / 0.5 / 0:

| Model | MSE | Δ |
|---|---|---|
| Plain Elo: no home advantage, margin, damping or carry-over | 0.16882 | — |
| ...with home advantage | 0.16217 | −3.9% |
| ...with margin of victory | 0.16208 | −0.06% |
| ...with autocorrelation damping | 0.16173 | −0.22% |
| ...with season carry-over (**as shipped**) | **0.16169** | −0.02% |

**Home advantage is the only one that matters.** The other three are worth a
few parts in a thousand each. They are kept because each is about three lines
and every one moved the number in the right direction — not because they are
important.

### Shipped constants

| | Value | Convention | Note |
|---|---:|---:|---|
| `k` | 14 | 20 | Surface is nearly flat between them |
| `home_advantage` | 80 | 60–100 | Worth 3.9% on its own |
| `season_carry` | 0.97 | 0.75 | **The convention is measurably too aggressive** |
| `initial` | 1500 | 1500 | |
| `scale` | 400 | 400 | 10:1 odds per 400 points |
| `damping` | 2.2 | 2.2 | FiveThirtyEight's constant |

They come from one pooled grid search over matches **before 2005-07-01**,
evaluated on everything after — a window chosen so the constants cannot encode
anything about the period they are judged on.

The `season_carry` result is the interesting one. Football-Elo implementations
conventionally regress 25% of a club's deviation away each summer. Three
decades of results say that is far too much: a club's strength persists across
a transfer window much more than the convention assumes, and 0.75 scores 0.16185
against 0.16162 for 0.95.

### What did not work

**Per-competition parameter fitting, which the plan called for.** Built,
measured, removed.

| Approach | MSE after the calibration window |
|---|---|
| Fixed constants | **0.16246** |
| Fitted per country on its first 5 seasons | 0.16279 |
| Fitted per country on seasons 3–8 (warmer window) | 0.16300 |
| Fitted on all countries pooled, pre-2005 | 0.16236 |

Per-country fitting made the ratings **worse**, and a warmer window made it
worse still. The reason is structural: a calibration window is the coldest part
of the history — every team starts on the same rating, so the window is mostly
warm-up noise, and three free parameters happily chase it. Pooling across
countries was a wash (0.16236 against 0.16246, which is 0.06%).

The surface is flat. That is the finding, and it is why the pipeline tunes
nothing at runtime. `scripts/build_ratings.py --fit-until DATE` re-derives the
constants when the data grows.

Per-competition tuning may still pay against a **three-class** objective, which
Elo alone cannot express — that belongs to Milestone 7, which owns the splits
such a fit would need.

### What Elo does not give you

`elo_expected_home` is an expected **score** on the 1 / 0.5 / 0 scale, not a
probability of a home win. It mixes a win and a draw and cannot be split into
three class probabilities without a further model. It is named for what it is,
because a column called `elo_prob_home` would be misread by everyone who
touched it.

---

## Dixon-Coles

Elo answers "who is stronger". This answers "how many goals, to whom, with what
probability" — and unlike Elo it produces a genuine H/D/A distribution, which
makes it the first thing in this project that can be scored with log loss and
compared against a bookmaker.

    lambda = exp(attack_home + defence_away + home_advantage)
    mu     = exp(attack_away + defence_home)

Home goals Poisson(lambda), away goals Poisson(mu), with two departures from
independence doing the real work: a one-parameter correction on the four
lowest scorelines, and exponential time decay over the fitting window.

Strengths are fitted **per competition** — attack and defence are only
identifiable among teams that actually play each other — and refitted as the
seasons advance.

### How good is it

English Premier League, 12,265 priced matches:

| | log loss | RPS |
|---|---|---|
| Class prior — predict 45 / 27 / 28 every time | 1.0647 | 0.2267 |
| **Dixon-Coles** | **0.9888** | **0.2008** |
| Bookmaker closing odds, overround removed | 0.9631 | 0.1944 |

The closing line is the strongest public forecast there is, and it is 0.026 of
log loss ahead. That gap is the honest size of the problem, and this is a
*rating* — the model zoo has not started yet.

### The settings, and what each is worth

| | Shipped | Convention | Measured |
|---|---:|---:|---|
| `decay` | 0.002 /day | ~0.0065 | Half-life 347 days. 0 → 0.9942, 0.003 → 0.9931, 0.0065 → 1.0004, 0.012 → 1.0182 |
| `window_days` | 1095 | 730 | 365 → 1.0051, 730 → 0.9925, 1095 → 0.9908, 1460 → 0.9902 |
| `refit_days` | 60 | — | 30 → 0.9908, 60 → 0.9888, 90 → 0.9947 |
| `min_matches` | 150 | — | Below this the window cannot identify two strengths per team |

Three findings worth stating plainly.

**The conventional decay is too fast.** Dixon and Coles' half-life of roughly
half a year loses to one of nearly a full season on this data — and *some*
decay still beats none, so the parameter earns its place; it is only the usual
value that does not.

**A longer window keeps winning.** Three seasons beats two, and four beats
three by 0.0006 — which is where it stops being worth a third more arithmetic.
The decay already handles recency, so the window's job is not to be recent: it
is to contain enough fixtures to identify the strengths of teams that rarely
meet.

**Sixty days between refits is better *and* half the cost.** The differences
between 30, 60 and 90 are within noise on subsets that differ by 2% — a longer
gap leaves a few more early matches unpriced — and when a metric cannot
separate two options, runtime can. This halves a full build from about twenty
minutes to ten.

### The low-score correction is worth almost nothing

The correction is the model's defining feature: independent Poissons
under-count 0-0, 1-0, 0-1 and 1-1, and one parameter reweights exactly those
four cells.

| | log loss | RPS |
|---|---|---|
| Correction off (rho pinned to 0) | 0.9910 | 0.2012 |
| Correction on | 0.9908 | 0.2011 |

**0.0002 of log loss.** It is kept — it is four lines, it is fitted rather than
assumed, and it is in the right direction — but anyone reaching for
Dixon-Coles over a plain double Poisson because of it should know the size of
what they are buying. At an earlier decay setting it measured slightly
*negative*, which is a fair summary of how large the effect is.

### Causality, and the case the generic probes cannot reach

The refit window ends **strictly before** the date of the match that triggered
it. That `<` rather than `<=` is the whole design, because a full Saturday
programme is one round: a model fitted on the 3pm results to predict the 5.30
kick-off would look excellent and be useless, and the leak is invisible to
truncation because both matches survive or neither does. There is a test for
exactly that.

Everything before a competition's first viable fit is null, and so is a team
promoted into a division that the current window has never seen. Both are
honest answers rather than a number nobody could have had. Coverage over the
full ingest is about 95%.

---

## Causality

Ratings are the first thing here with memory, and therefore the first place a
leak can hide. Reading the code for it does not scale: a rolling mean with an
off-by-one window, a normalisation over the whole table, a rating updated
before it is read — all look correct and all leak.

So the property is tested, by two probes in `src/validation/temporal.py` that
know nothing about how a value was produced:

- **Prefix invariance** — truncate the input after match *n*, recompute, and
  every surviving row must be byte-identical.
- **Outcome independence** — rewrite one match's scoreline; rows up to and
  including that match must not move.

**Both are needed, and this is measurable rather than asserted.** A derivation
that reads its own row is *perfectly* prefix-invariant — every value only ever
depended on its own row, so truncation changes nothing — and is caught only by
the rewrite:

| Planted leak | Prefix invariance | Outcome independence |
|---|---|---|
| Mean over the whole table | **caught** | **caught** |
| Reads its own match's result | passes | **caught** |
| Lagged by one match (causal) | passes | passes |

Elo passes both over the full table. The probes are generic over
`Callable[[DataFrame], DataFrame]`, so Milestone 6 runs the same two over
feature builders rather than writing a second suite.
