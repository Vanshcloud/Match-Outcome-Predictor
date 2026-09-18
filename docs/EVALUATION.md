# Evaluation

Walk-forward folds, four baselines, two proper scoring rules, and the number
that says how much of the problem is left.

---

## The short version

Five folds of a year each, ending with the most recent match in the table.
62,036 evaluation matches, scored on the 59,001 that every forecaster could
price:

| | log loss | RPS | accuracy |
|---|---:|---:|---:|
| Bookmaker closing odds, overround removed | **0.9993** | **0.2031** | 50.5% |
| Dixon-Coles | 1.0277 | 0.2114 | 48.5% |
| Class prior, counted per fold | 1.0751 | 0.2284 | 43.7% |
| Home always | ∞ | 0.4316 | 43.7% |

**The gap that matters is 0.0284 of log loss** — Dixon-Coles against the
closing line. That is the honest size of what the model zoo has to close, and
it is measured on identical matches rather than on two convenient subsets.

The rating is worth having: 0.047 of log loss clear of the prior, which is the
floor a model that has learned nothing lands on. And the bookmaker wins
everywhere, in all 39 competitions, which is the result to expect. A model that
beat the closing line on public data would be evidence of a leak rather than of
skill.

**Where that 0.0163 lives** is measured rather than guessed:
grouped by how far the model was from the closing line, the deficit runs from
+0.0009 where the two agree to +0.1835 where they are furthest apart. See
[Where the gap to the closing line actually lives](#where-the-gap-to-the-closing-line-actually-lives).

---

## How the folds are cut

Expanding window, anchored at the end of the history:

| Fold | Trains to | Scores (end exclusive) | Matches |
|---|---|---|---:|
| 0 | 2021-09-02 | 2021-09-03 → 2022-09-03 | 13,030 |
| 1 | 2022-09-02 | 2022-09-03 → 2023-09-03 | 12,302 |
| 2 | 2023-09-02 | 2023-09-03 → 2024-09-02 | 12,503 |
| 3 | 2024-09-01 | 2024-09-02 → 2025-09-02 | 12,399 |
| 4 | 2025-09-01 | 2025-09-02 → 2026-09-02 | 11,802 |

**A year, not a season.** The 39 competitions here start in August, in January
and in April, so "the season" is a boundary drawn in a different place for each
of them and not a boundary at all for the pooled table.

**Expanding, not sliding.** There is no reason to throw away 1993 when scoring
2024. Recency is already handled inside the rating, by a time decay whose
half-life was measured rather than assumed; a sliding window would be a second,
blunter opinion about the same thing.

**The cut is a date.** A boundary at row *n* puts two matches played on the
same afternoon on opposite sides of it, ordered only by whatever tiebreak the
sort used — so the model trains on the 3pm results and is scored on the 5.30
kick-off. The same rule the feature windows follow, for the same reason.

**Every fold is probed, not trusted.** `split_boundary` from
[LEAKAGE.md](LEAKAGE.md) runs on each fold before it is used: no training row
dated at or after any evaluation row, no match in both halves, ties failing.
The arithmetic that produces the folds is four lines and obviously correct,
which is exactly the kind of code that is quietly wrong after someone changes a
`<` to a `<=` for a reason that made sense at the time.

---

## The four baselines

**Home always.** Certainty that the home side wins. It is right 43.7% of the
time — which is why accuracy is the wrong target — and it scores **infinite**
log loss, which is not clipped. A forecast that ruled out what happened was
infinitely wrong, and reporting that as some large finite number would be a
kindness the metric does not extend. RPS still ranks it at 0.4316, because RPS
is bounded and knows the classes are ordered. That contrast is the whole reason
both metrics are here.

**Class prior.** The base rates of home, draw and away, **counted on the
training fold** and applied to every match in the next one. This is the floor: a
model that cannot beat it has learned nothing about *which* match it is looking
at. Counted per fold rather than taken from a constant, because a prior read
off the whole table has seen the future — and measurably so, since the home-win
rate has drifted over the thirty years in this dataset.

**Dixon-Coles.** The goal-rate rating, read from the ratings table. The rating
was fitted over the whole history, which sounds like a leak and is not: every
row of it depends only on matches strictly earlier than its own, and that is
the property the temporal probes establish on every build. Recomputing it per
fold would produce the same numbers at ten times the cost.

**Bookmaker.** The closing line with the overround removed by dividing each
implied probability by their total. Not the only way to de-vig — the favourite
carries more of the margin than an equal share — but the transparent one. A
cleverer removal would make the benchmark a modelling choice rather than a
measurement.

---

## Why there are two tables

The bookmaker prices 99.8% of these matches and Dixon-Coles prices 95.3%, and
those are not the same matches. So every forecaster is scored twice:

| Over the matches it could price | coverage | n | log loss | RPS |
|---|---:|---:|---:|---:|
| bookmaker | 99.8% | 61,889 | 0.9999 | 0.2032 |
| dixon_coles | 95.3% | 59,148 | 1.0275 | 0.2114 |
| class_prior | 100% | 62,036 | 1.0751 | 0.2283 |
| home_always | 100% | 62,036 | ∞ | 0.4316 |

Only the **common** table at the top of this page may be read across rows. This
one may not: each figure answers a slightly different question, and the
differences between them are the same size as the differences the project is
trying to measure. Dixon-Coles cannot price a competition's first seasons — it
needs a window to fit on — which is where its missing 4.7% goes.

---

## Fold to fold

| Fold | n | bookmaker | dixon_coles | class_prior |
|---|---:|---:|---:|---:|
| 0 | 12,355 | 0.9978 | 1.0297 | 1.0773 |
| 1 | 11,672 | 0.9962 | 1.0225 | 1.0723 |
| 2 | 11,904 | 0.9973 | 1.0307 | 1.0766 |
| 3 | 11,829 | 1.0016 | 1.0251 | 1.0740 |
| 4 | 11,241 | 1.0042 | 1.0303 | 1.0751 |

Stable: the rating moves over a range of 0.008 and the line over 0.008, with
the gap between them never leaving 0.021–0.028. Nothing here is a lucky year,
which is the first thing a single-split number cannot tell you.

---

## By competition

Dixon-Coles beats the class prior in **38 of 39** competitions, and loses to
the closing line in all 39. The gap to the line ranges from 0.014 to 0.081.

### The rating's edge is the spread of team strength, at r = 0.90

How much Dixon-Coles beats the prior by, per competition, against the standard
deviation of Elo across the sides in that competition's recent matches:

| | Elo spread | prior − Dixon-Coles |
|---|---:|---:|
| GRE_1 | 134.0 | 0.130 |
| POR_1 | 132.8 | 0.147 |
| NED_1 | 127.8 | 0.114 |
| SCO_1 | 121.0 | 0.114 |
| … | | |
| ENG_2 | 54.4 | 0.020 |
| FRA_2 | 52.5 | 0.015 |
| SCO_2 | 50.7 | 0.013 |
| ESP_2 | 48.4 | 0.018 |

**Correlation 0.90 over 39 competitions.** A strength model has the most to say
where strengths differ most, which is exactly what it should do and is not
something anyone told it to do. A Portuguese or Greek season is two or three
serious clubs and a long tail; an English or Spanish second division is twenty
clubs within fifty Elo points of each other, and there the prior is nearly as
good as anything.

The bookmaker's edge over the prior tracks the same thing at r = 0.89, so this
is a property of the competitions rather than of the model.

### What is left to win is not strength

The gap between Dixon-Coles and the closing line correlates with strength
spread at **−0.14** — that is, not at all. The bookmaker's advantage is roughly
the same 0.028 whether the league is lopsided or level.

That is the most useful number in this document for what comes next. The
rating has already extracted what team strength can explain; whatever the
closing line knows on top of it is something else — form, availability,
motivation, and the market's own information — and it is worth about the same
amount everywhere. The model zoo was aimed at that gap; [MODELS.md](MODELS.md)
reports how much of it six families, a blend and a calibration layer closed —
0.0121 of 0.0284, with the rest still looking like information rather than
capacity.

### Where the gap is widest, it is where the data is thinnest

The claim that what remains is missing information rather than missing
capacity is made three times in this repository. It is checkable, and this is
the check: split the 39 competitions by whether they carry shot statistics at
all, and compare the shipped model's gap to the closing line on each side.

| | competitions | matches | gap to the closing line |
|---|---:|---:|---:|
| Carries shot data | 10 | 19,903 | **0.0134** |
| Carries none | 29 | 39,098 | **0.0177** |

The gap is **32% wider** where the provider publishes results and odds and
nothing else. Across the 39 per-competition gaps the correlation with shot
coverage is **−0.36**, and a Welch t-test on the two groups gives
**t = −2.08, p = 0.045**.

That is the predicted direction and a real effect, and it is worth being
precise about how strong it is not. Ten competitions against twenty-nine is a
small side, p = 0.045 clears the conventional bar and no more, and the split
is confounded: the shot-capable competitions are also the long-running
European leagues with the deepest history and the most bookmaker attention, so
"has shot data" is partly a proxy for "is a league the market prices
carefully". The honest reading is that the evidence points the way the
argument does, not that it settles it.

What would settle it is a competition that gains shot coverage mid-history,
scored either side of the change. Two of the extra-schema files are close to
that shape and none quite is.

### The one it loses

| | n | bookmaker | dixon_coles | class prior |
|---|---:|---:|---:|---:|
| ARG_CUP | 569 | 1.0589 | **1.1329** | 1.0902 |

**Dixon-Coles is worse than counting base rates on the Argentine cup.** The
obvious explanation — a cup draws clubs from different tiers, so the model is
extrapolating — is wrong, and the data says so: all 32 clubs in the recent cup
also play in ARG_1, and the competition's Elo spread is 55.5, the seventh
*narrowest* of the 39. Two things are true instead, and they compound. The
sides are closely matched, so there is little for a strength model to add; and
the model refits **per competition**, so the cup's fit sees 610 cup matches
while the same clubs' 2,211 league matches sit next door, unused.

It is left in rather than special-cased. A per-competition exclusion has to be
justified for every competition it is *not* applied to. What the finding
actually argues for is pooling a cup's fit with its country's league, which is
a change to the rating and needs an ablation to justify it.

---

## Where the gap to the closing line actually lives

The project-level number — the shipped model 0.0163 behind the bookmaker over
the 59,001 matches every forecaster could price — is a mean, and a mean can hide
two very different worlds: a model uniformly a little worse everywhere, or a
model level with the line on most fixtures and badly wrong on some. It is the
second.

The table below drops only what *one of these two* could not price, so it runs
over 61,889 matches rather than 59,001 — 2,888 wider, because it does not also
require Dixon-Coles to have priced the match. The deficit over its own
population is 0.0168.

Every out-of-sample forecast, grouped by how far it was from the closing line.
"Apart" is total-variation distance, which for three outcomes reads as a
percentage-point gap — 0.07 is "seven points apart":

| Apart | n | Model | Market | Model − market | Model better |
|---|---:|---:|---:|---:|---:|
| <2% | 9,628 | 0.9881 | 0.9872 | **+0.0009** | 49.2% |
| 2–5% | 21,139 | 1.0136 | 1.0099 | +0.0037 | 48.4% |
| 5–10% | 20,874 | 1.0199 | 1.0059 | +0.0140 | 46.4% |
| 10–20% | 9,419 | 1.0402 | 0.9864 | +0.0538 | 41.9% |
| >20% | 829 | 1.0810 | 0.8975 | **+0.1835** | 35.5% |

`make card` writes this to `data/reports/ensemble/market.parquet` from
`src/pipelines/report.py::market_comparison`, over the same per-match forecasts
the reliability tables are computed from.

**Where the model agrees with the line, it is level with it.** +0.0009 over
9,628 matches is not a deficit worth a sentence. The blend and calibration
bought 0.0003 of the 0.0165 the model zoo left; this says the remaining gap
is not spread thinly across every match, it is concentrated in the ones the
model sees differently.

**The disagreement predicts the model's error, not the market's.** The deficit
grows by a factor of about 200 from the narrowest band to the widest, and in
that widest band the market's own log loss *improves* to 0.8975 — those 829
matches are ones it prices confidently and correctly while the model does not.
The share of matches the model scores better on falls monotonically, 49.2% to
35.5%.

That rules out the reading a value detector rests on. "The model says 45% and
the price says 38%, so there is value in the difference" is a testable claim,
and the test is above: the wider that difference, the more likely it is that
the model is the one that is wrong.

**What this is not.** It is not a claim that a model *cannot* beat a closing
line, and it is not a betting result: log loss is a scoring rule, not a profit
and loss, and none of these figures accounts for the margin that was removed to
compute them. It is a statement about this model against this line over these
folds.

---

## What is measured, and what is not

**Log loss** is the likelihood of what happened. It is the metric the
bookmaker's line is usually quoted in and the one the project states its
ceiling in, and it punishes confidence without mercy.

**RPS** knows H, D and A are ordered. Predicting a draw when the away side won
is a smaller error than predicting a home win, and log loss cannot see the
difference — both ruled out what happened, so both score infinity. RPS is also
what ranks a degenerate forecast that log loss has already sent to infinity.

**Accuracy** is reported last because in this problem it is close to
meaningless. Predicting home every time scores 43.7%; the closing line scores
50.5%. A well-calibrated model that rarely *predicts* a draw is behaving
correctly, not failing — roughly a quarter of matches are drawn and almost none
of them are the modal outcome beforehand.

**Reliability** — whether a stated probability happens as often as it says —
lives in `src/evaluation/reliability.py`, and is reported
in [MODELS.md](MODELS.md) rather than here: everything on this page is a
baseline, and the two that are fitted are fitted on likelihood already.

**Per-class and per-competition breakdowns** are not here either. They belong
with the model card rather than with the baselines, and they are in
[MODEL_CARD.md](MODEL_CARD.md) — where the draw column turns out to be the most
reliable and the least useful, and the Argentine cup the least reliable of the
39.

---

## What the service actually served

Everything above is the *backtest* — a measurement of a model on
folds cut out of history. This section is about the deployment: the forecasts
the service really answered, scored against the results that arrived afterwards.

`make archive` produces it, from the prediction log rather than from any table
on disk, and the comparison it draws is the only definition of drift this
project uses: served log loss against the walk-forward figure for the same
model. Not a distance between feature distributions — that measures that an
input moved, which is a hypothesis about performance rather than a measurement
of one.

### The sample size is the finding

Per-match log loss over the 62,036 walk-forward forecasts has a mean of 1.0165
and a **standard deviation of 0.3976**. That spread is what a served mean has to
be read against:

| Scored forecasts | Smallest shift distinguishable from noise |
|---:|---:|
| 8 | 0.2755 |
| 100 | 0.0779 |
| 1,000 | 0.0246 |
| 10,000 | 0.0078 |
| 62,036 | 0.0031 |

Run backwards: **2,286** scored forecasts to see a shift the size of the 0.0163
this project's whole argument is about, 243 to see 0.05, 6,073 to see 0.01.
These are the half-width of a two-sided 95% interval, which catches a real
shift of exactly that size about half the time. Catching it 80% of the time
takes about twice as many: **4,670** for 0.0163.

So "no drift detected" is almost always the wrong sentence. The right one is
"this archive could not detect a shift smaller than X", and `detectable` is a
column of the report rather than a caveat somebody has to remember.

### Checked against the real tables

The scoring path was driven with the walk-forward forecasts standing in for
served ones — the only out-of-sample forecasts this project has:

| Rows | Served | Drift | Detectable | Reported as |
|---:|---:|---:|---:|---|
| 100 | 1.1048 | +0.0883 | 0.0779 | distinguishable |
| 3,000 | 1.0286 | +0.0122 | 0.0142 | **not** distinguishable |
| 62,036 | 1.0165 | +0.0000 | 0.0031 | not distinguishable |

The last row is the correctness check: fed the same forecasts, the served path
and the backtest path agree to four decimals.

The first row is worth reading carefully, because it is not a false positive of
the test — it is a selection effect. The first hundred rows of that table are
the first hundred matches of fold 0, not a random sample of it, and a
non-random hundred matches is exactly the shape a young archive has: one
weekend, a few competitions, whatever the service happened to be asked about.
**A threshold cannot rescue a sample that is not random**, which is a second
reason not to read a small archive as drift.

### Why the archive had nothing in it, and what changed

The first archive run found 35 rows in the log, 25 after repeats collapsed, **25 of them
in-sample and 0 scorable**, and concluded that this was a property of how the
service was being driven rather than of the code. It was, and the property had
a name: this project ingests results, so every fixture the service could be
asked about was one the shipped artefact had trained on. `in_sample` was
`true` on all of it because it could not have been anything else.

The fix is two commands rather than a change to any
measurement on this page:

```bash
make fixtures   # design rows for matches that have not been played, ~12 min
make price      # ask the service about them, so the log fills before kick-off
```

`make fixtures` appends the published fixture list to the canonical frame and
runs the same ratings and feature builders over the whole thing — the design
row a fixture gets is the one `make features` writes for it once it is played,
checked by holding out the last day of the real table and rebuilding it as a
fixture: **every one of the thirty columns agreed to zero.** `make price` then
asks the running service for those fixtures, and because they are dated after
the artefact's training window, the service stamps them `in_sample: false`.

So the archive's exclusion counts are no longer structural. What is left is the
sample size, which is the finding above and is a matter of weeks rather than of
code: at this project's own gap to the closing line, a verdict needs **2,286
scored forecasts**, and a fixture list is a few hundred a week.

---

## Re-running it

```bash
make backtest                      # 5 folds of a year each, ~5 seconds
python scripts/backtest.py --folds 10 --horizon-days 182
python scripts/backtest.py --competition ENG_1
python scripts/backtest.py --markdown        # the tables above
make card                          # also writes the disagreement table, ~5 min
make archive                       # scores the prediction log; needs PREDICTION_LOG_DSN
```

The finest grain — one row per fold, per competition, per forecaster, per
subset — is written to `data/reports/backtest.parquet` with a manifest beside
it, whose `inputs` block records the SHA-256 of the match, ratings and feature
tables the scores came from. Every metric is a mean over matches, so any coarser view is a weighted mean
of those rows and reconstructs exactly what a direct pass would have produced.
That is why the pooled figures are computed rather than scored a second time.
