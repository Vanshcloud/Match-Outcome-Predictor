# Models

Six families over thirty columns, scored on the folds Milestone 7 built and
against the baselines it measured. What each one is worth, what tuning bought,
and what each feature block contributes.

---

## The short version

Five folds of a year each, 62,036 evaluation matches, scored on the 59,001 that
every forecaster could price:

| | log loss | RPS | accuracy |
|---|---:|---:|---:|
| Bookmaker closing odds | **0.9993** | **0.2031** | 50.6% |
| CatBoost | **1.0159** | 0.2083 | 49.3% |
| XGBoost | 1.0161 | 0.2084 | 49.3% |
| LightGBM | 1.0161 | 0.2083 | 49.3% |
| Logistic regression | 1.0162 | 0.2083 | 49.3% |
| Random forest | 1.0172 | 0.2087 | 49.2% |
| MLP | 1.0203 | 0.2091 | 49.1% |
| Dixon-Coles (Milestone 4) | 1.0277 | 0.2114 | 48.5% |
| Class prior | 1.0751 | 0.2284 | 43.7% |
| Home always | ∞ | 0.4316 | 43.7% |

**Every family beats the rating it was built on**, in all 39 competitions. The
best of them closes **0.0118 of the 0.0284** Milestone 7 measured between
Dixon-Coles and the closing line — 42% of the gap — and leaves 0.0166.

And the finding that matters more than the ranking: **the top four are within
0.0003 of each other.** Logistic regression, on thirty columns, is not
distinguishable from three tuned gradient-boosting libraries. Milestone 7 said
what the bookmaker knows on top of a strength rating is worth about the same
amount in every competition and is not strength; this says it is not a function
these thirty columns express non-linearly either. Whatever is left is missing
information, not missing model capacity.

---

## What a model sees

Thirty columns: the twenty features from Milestone 5 and the ten rating columns
from Milestone 4.

| Block | Columns | What it is |
|---|---:|---|
| `form` | 14 | Points, goals and shots for and against over the previous five, plus venue form and a match count per side. |
| `schedule` | 4 | Rest days and matches in the previous fortnight, per side. |
| `head_to_head` | 2 | Previous meetings, and the home side's record in them. |
| `elo` | 5 | Two ratings, the expectation between them, and the matches behind each. |
| `dixon_coles` | 5 | Two scoring rates and the three-class probabilities they imply. |

The list is **derived** from the two registries, not written out again. A
feature added in `src/feature_engineering/registry.py` is a column the zoo sees
without anyone editing a second list, and a column that stops existing stops
being requested — which is the failure a hand-maintained list produces a
milestone later, as a table of nulls nobody notices.

Nothing canonical is passed through directly. Not the scoreline, obviously.
**Not the odds either**: they are pre-match and still withheld, because they
are the benchmark this project measures itself against and a model given the
closing line learns to copy the bookmaker. Both exclusions are set
intersections in the test suite rather than a rule someone has to remember.

---

## How a model is fitted

A trained model is a `Forecaster` — the same shape the baselines already
satisfy — that fits inside its own `forecast(train, evaluate)`. Three things
follow, and all three are the point:

**The evaluation pipeline is unchanged.** Nothing in `src/pipelines/backtest.py`
knows an estimator exists. A model and a baseline are scored by the same code,
on the same folds, in the same two subsets, into the same file format — which
is the only reason their numbers can be put in one table.

**Causality is structural, not checked.** A model cannot reach a match it was
not handed, and the split layer decides what it is handed. There is no
derivation to recompute here, so the prefix and outcome probes have nothing to
probe; what *is* checked, per family, is that rewriting the evaluation half's
results does not move the forecast.

**A fresh estimator per fold.** `build` is a callable, not an instance. An
instance reused across folds would carry fold 0's fit into fold 1 — a leak with
no symptom at all, because the scores would simply come out better than they
should.

### Two smaller decisions

**Class order is fixed by encoding.** Targets are integer indices in H, D, A
order, so `predict_proba` comes back in the order the metrics expect. Handing
scikit-learn the labels would sort them alphabetically — A, D, H — and
transpose every forecast into something that still sums to one and scores
plausibly.

**Nulls are handled where the choice is visible.** A null means "no history
yet": a club's first matches, a competition's first seasons. The three boosted
families read that directly and do better for it. The three that cannot —
logistic regression, the forest, the MLP — get a median imputer in their own
pipeline, so the substitution is a line of code rather than a pre-filled
matrix.

---

## Tuning, on matches the report never sees

The obvious way to tune is to optimise the number the report prints. It is also
the way to publish a number that will not survive next season. So the search
runs on a **tuning slice**: every match strictly earlier than the first
reported fold — 241,481 of them, ending 2021-09-02 — with the same walk-forward
arrangement inside it. Nothing the search touches is scored later.

The winners are then pasted into `src/models/zoo.py` as documented constants,
the way Dixon-Coles' decay and window are. A tuned parameter living in a JSON
file beside the data is a parameter regenerated on one machine and stale
everywhere else.

### What the search bought

Log loss on the tuning folds, best against the value the search started from.
The budgets differ, and are stated rather than implied: a trial is three fits
over most of the history, and the whole set took about forty minutes.

| | trials | best | started from | bought |
|---|---:|---:|---:|---:|
| MLP | 3 | 1.0272 | 1.0416 | **+0.0144** |
| LightGBM | 8 | 1.0218 | 1.0246 | **+0.0028** |
| XGBoost | 20 | 1.0213 | 1.0224 | +0.0010 |
| CatBoost | 8 | 1.0216 | 1.0218 | +0.0001 |
| Random forest | 6 | 1.0225 | 1.0226 | +0.0001 |
| Logistic regression | 20 | 1.0219 | 1.0219 | +0.0000 |

**Four of the six searches moved the fourth decimal place.** Twenty trials of
logistic regression could not beat `C=1.0` at all — with 241,000 rows and
thirty columns the penalty is irrelevant, and the search discovering that is a
useful thing to have discovered rather than assumed.

The two that did move are both cases of a bad starting guess rather than a
subtle optimum. LightGBM's defaults were too coarse for this problem. The MLP's
were simply wrong: the search preferred one hidden layer of 64 to two of 64 and
32, and preferred it by 0.0144 — which is a fair measure of how wrong an
untested guess at an architecture can be, and it is *still* the worst family
here afterwards.

The tree search spaces were narrowed once, after measurement. LightGBM's first
version reached 255 leaves and 800 estimators, and one trial in that corner
took longer than the entire XGBoost search — for a model nobody would ship on
thirty tabular columns. Every number in the table above used the space as it
now stands.

---

## What each feature block is worth

LightGBM, each block withheld in turn, all six variants scored **in one
backtest** so they share a common subset. Six separate runs would each compute
their own, and the differences between those subsets are the same size as the
differences being measured.

| Block withheld | log loss | Δ log loss | Δ RPS |
|---|---:|---:|---:|
| nothing (control) | 1.0169 | — | — |
| `form` | 1.0201 | **+0.0033** | +0.0010 |
| `elo` | 1.0189 | +0.0021 | +0.0006 |
| `dixon_coles` | 1.0185 | +0.0016 | +0.0005 |
| `schedule` | 1.0172 | +0.0003 | +0.0001 |
| `head_to_head` | 1.0169 | +0.0001 | +0.0000 |

**Form is worth more than either rating.** Fourteen columns of rolling windows
over a sorted array beat a bivariate Poisson refitted every sixty days and an
Elo pool per country — each taken on its own. That is not an argument against
the ratings; they are substitutes for each other, and the model can recover
most of what one says from the other. It is an argument about where the
remaining effort goes.

**Rest days and congestion are worth 0.0003.** Milestone 5 shipped them saying
they carried no marginal signal and that this ablation would settle it. It has.
They survive — four columns, computed by the same pass that computes the rest,
and a positive delta is a positive delta — but they are the first thing to go
if the feature set ever needs trimming, and nothing should be built on top of
them.

**Head-to-head is worth 0.0001**, which is nothing. Two teams' previous
meetings tell a model what their strengths already told it.

### The competition the rating could not price

Milestone 7 found Dixon-Coles losing to *counting base rates* on the Argentine
cup: 1.1329 against 1.0902, the only competition of 39 where that happened. It
diagnosed a narrow field fitted per competition on 610 matches while the same
clubs' 2,211 league matches sat unused next door, and asked whether the answer
was to pool the cup's fit with its league.

The answer is that no change to the rating is needed:

| ARG_CUP, 569 matches | log loss |
|---|---:|
| Bookmaker | 1.0589 |
| **LightGBM** | **1.0806** |
| Class prior | 1.0902 |
| Dixon-Coles | 1.1329 |

The model has somewhere else to look. Where the rating is unidentified it
leans on form, and the result moves from 0.043 *worse* than the prior to 0.010
better — without a special case, a per-competition rule, or anyone telling it
which competition was the awkward one. That is the argument for a model over a
rating, stated in one competition.

---

## By competition

LightGBM beats Dixon-Coles in **39 of 39** competitions and the class prior in
all 39; the bookmaker beats it in all 39. The gap to the closing line ranges
from 0.0059 to 0.0615, with a median of 0.0154 — against Dixon-Coles' median of
0.0263 at the same point in Milestone 7. The zoo took about 40% of the gap in
the median competition, which is the same share it took overall.

The widest remaining gap is the Chinese Super League at 0.0615, and the
narrowest is the Russian Premier League at 0.0059. Neither is a competition
this project has any reason to model differently; what the spread says is that
the market's advantage is not uniform, only that it does not track team
strength — which was Milestone 7's finding and survives the zoo.

---

## What is not here

**No ensemble, and no calibration.** Both are Milestone 9. Averaging the top
four would be the obvious next move and it is deliberately not taken here — the
four are within 0.0003 of each other and heavily correlated, so an ensemble
built now would report a number that says more about the averaging than about
the models.

**No feature added to close the gap.** The ablation says where the remaining
value is not, and Milestone 7 says what is left is not team strength. Choosing
what to add on that evidence is worth doing carefully rather than immediately.

**No per-competition model.** The Argentine cup result above is the case that
would have justified one, and it turned out not to need it.

---

## Re-running it

```bash
make train                                   # six families, five folds (~10 min)
make ablation                                # what each block is worth
python scripts/train.py --model lightgbm
python scripts/train.py --tune xgboost --trials 20
python scripts/train.py --markdown           # the tables above
```

The score table goes to `data/reports/zoo/backtest.parquet` and the ablation to
`data/reports/ablation/backtest.parquet`, both written by the unchanged
evaluation pipeline with a manifest beside them. Runs are logged to a local
SQLite MLflow store under `models/` — not a server, and not MLflow's own
directory store, which 3.x refuses to write to at all.
