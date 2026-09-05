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
| Bookmaker closing odds, overround removed | **0.9993** | **0.2031** | 50.6% |
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

---

## How the folds are cut

Expanding window, anchored at the end of the history:

| Fold | Trains to | Scores | Matches |
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

**Dixon-Coles.** Milestone 4's rating, read from the ratings table. The rating
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
a change to the rating and belongs in the milestone that has an ablation to
justify it with.

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
50.6%. A well-calibrated model that rarely *predicts* a draw is behaving
correctly, not failing — roughly a quarter of matches are drawn and almost none
of them are the modal outcome beforehand.

**Reliability** — whether a stated probability happens as often as it says —
arrived with Milestone 9, in `src/evaluation/reliability.py`, and is reported
in [MODELS.md](MODELS.md) rather than here: everything on this page is a
baseline, and the two that are fitted are fitted on likelihood already.

**Per-class and per-competition breakdowns** are not here either. They belong
with the model card rather than with the baselines, and they are in
[MODEL_CARD.md](MODEL_CARD.md) — where the draw column turns out to be the most
reliable and the least useful, and the Argentine cup the least reliable of the
39.

---

## Re-running it

```bash
make backtest                      # 5 folds of a year each, ~5 seconds
python scripts/backtest.py --folds 10 --horizon-days 182
python scripts/backtest.py --competition ENG_1
python scripts/backtest.py --markdown        # the tables above
```

The finest grain — one row per fold, per competition, per forecaster, per
subset — is written to `data/reports/backtest.parquet` with a manifest beside
it. Every metric is a mean over matches, so any coarser view is a weighted mean
of those rows and reconstructs exactly what a direct pass would have produced.
That is why the pooled figures are computed rather than scored a second time.
