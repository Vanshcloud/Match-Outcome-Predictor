# Models

Six families over thirty columns, a blend of the three that are not
substitutes, and a calibration layer over both — all scored on the
walk-forward folds and against the baselines in EVALUATION.md. What each one is
worth, what tuning bought, what each feature block contributes, and what the
last two layers did not buy.

---

## The short version

Five folds of a year each, 62,036 evaluation matches, scored on the 59,001 that
every forecaster could price:

| | log loss | RPS | accuracy |
|---|---:|---:|---:|
| Bookmaker closing odds | **0.9993** | **0.2031** | 50.6% |
| Blend of three, calibrated | **1.0156** | **0.2082** | 49.3% |
| Blend of three | 1.0156 | 0.2082 | 49.3% |
| CatBoost | 1.0159 | 0.2083 | 49.2% |
| XGBoost | 1.0161 | 0.2084 | 49.3% |
| LightGBM | 1.0161 | 0.2083 | 49.3% |
| Logistic regression | 1.0162 | 0.2083 | 49.2% |
| Random forest | 1.0172 | 0.2087 | 49.2% |
| MLP | 1.0203 | 0.2091 | 49.1% |
| Dixon-Coles rating | 1.0277 | 0.2114 | 48.5% |
| Class prior | 1.0751 | 0.2284 | 43.7% |
| Home always | ∞ | 0.4316 | 43.7% |

**Every family beats the rating it was built on**, in all 39 competitions. The
zoo closes **0.0118 of the 0.0284** the baseline backtest measured between Dixon-Coles
and the closing line — 42% of the gap — and the blend and the calibration layer
together close **0.0003 more**, leaving 0.0163.

That last number is the honest headline of the ensembling work: averaging three
models whose errors are as uncorrelated as this zoo gets is worth a fortieth of
what the zoo itself was worth over the rating, and temperature scaling is worth
nothing at all in log loss. What calibration *does* buy is stated in its own
section below, and it is not a score.

And the finding that matters more than the ranking: **the top four are within
0.0003 of each other.** Logistic regression, on thirty columns, is not
distinguishable from three tuned gradient-boosting libraries. The baseline backtest showed
what the bookmaker knows on top of a strength rating is worth about the same
amount in every competition and is not strength; this says it is not a function
these thirty columns express non-linearly either. Whatever is left is missing
information, not missing model capacity.

---

## What a model sees

Thirty columns: the twenty registered features and the ten rating columns.

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
being requested — which is the failure a hand-maintained list produces
months later, as a table of nulls nobody notices.

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

**Rest days and congestion are worth 0.0003.** They were added on the
expectation that they carried no marginal signal, pending this ablation. It
settles it.
They survive — four columns, computed by the same pass that computes the rest,
and a positive delta is a positive delta — but they are the first thing to go
if the feature set ever needs trimming, and nothing should be built on top of
them.

**Head-to-head is worth 0.0001**, which is nothing. Two teams' previous
meetings tell a model what their strengths already told it.

### The competition the rating could not price

The baseline backtest found Dixon-Coles losing to *counting base rates* on the Argentine
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

## The blend, and why it is these three

Averaging models that are wrong about the same matches produces a model that is
wrong about those matches with slightly less confidence. No ensemble was
built on the zoo ranking alone for exactly that reason — its top four sat within 0.0003 of
each other — so the members here are chosen on **the correlation of their
per-match errors**, and their individual scores decide only the order
candidates are considered in.

Per-match log loss, correlated pairwise over 34,306 matches of the tuning slice
— every one of them earlier than the first reported fold, for the same reason
the hyperparameters were chosen there:

| | catboost | lightgbm | logistic_regression | mlp | random_forest | xgboost |
|---|---:|---:|---:|---:|---:|---:|
| catboost | 1.0000 | 0.9915 | 0.9872 | 0.9212 | 0.9922 | **0.9960** |
| lightgbm | 0.9915 | 1.0000 | 0.9826 | 0.9201 | 0.9860 | 0.9953 |
| logistic_regression | 0.9872 | 0.9826 | 1.0000 | 0.9231 | 0.9796 | 0.9857 |
| mlp | 0.9212 | 0.9201 | 0.9231 | 1.0000 | **0.9153** | 0.9213 |
| random_forest | 0.9922 | 0.9860 | 0.9796 | 0.9153 | 1.0000 | 0.9934 |
| xgboost | 0.9960 | 0.9953 | 0.9857 | 0.9213 | 0.9934 | 1.0000 |

The pairs are not evenly spread, and the gap in them is the whole decision.
Admission is greedy and best-first, so the column that decides it is the one
against **XGBoost**, the best family and therefore the first one in: the other
three trees sit at 0.9934 to 0.9960 against it — they are substitutes, which is
the quantitative form of the zoo's finding that the top four were
indistinguishable. (Among themselves the trees spread wider, down to 0.9860 for
lightgbm against random forest, but neither of those is ever in the blend to be
compared against.) Logistic regression is further out at 0.9857, and the MLP is
the only family wrong about different matches at all, at 0.92 against
everything — and it is also the worst model here.

`MAX_ERROR_CORRELATION = 0.99` sits in the empty band between 0.9857 and
0.9934 of that deciding column. It is read off the measurement rather than chosen in advance, and it
admits, best-first: **XGBoost, logistic regression, the MLP**.

| 59,001 matches | log loss | RPS |
|---|---:|---:|
| Blend of the three | **1.0156** | **0.2082** |
| CatBoost, the best single family | 1.0159 | 0.2083 |
| XGBoost, the blend's own best member | 1.0161 | 0.2084 |
| MLP, the blend's worst member | 1.0203 | 0.2091 |

**The blend is the best row — clearly against its members, within noise
against CatBoost.** It is 0.0005 of log loss better than XGBoost, its best
member, and 0.0003 better than CatBoost, which did not make it in. Paired over
the 193 fold × competition cells those gaps are about 2.4 and 1.4 standard
errors: a plain mean of a good model, a mediocre one and the worst one in the
zoo is measurably better than the models it averages and level with the best
single family. That is what choosing on error correlation is for, and it is the
argument against the obvious alternative of averaging the top three, all of
which correlate above 0.9915.

It is also very small. Per competition the blend beats XGBoost in 27 of 39 —
not 39 — and it beats Dixon-Coles and the class prior in all 39, as every
family already did.

**No fitted weights, and no log pool.** Weights fitted on any holdout small
enough to be honest are noise with three decimal places at this margin, and a
log pool sharpens a blend that the next section shows is already slightly too
confident.

---

## Calibration: it buys reliability, not loss

Temperature scaling: one scalar, `p ** (1/T)` renormalised, fitted on **the
last year of each fold's own training half** with the model refitted on
everything before it. The evaluation half is never read, which costs a second
fit per fold and is the only arrangement under which a calibrated model has
seen exactly the matches the uncalibrated one saw.

The fitted temperatures, per fold:

| Fold | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| XGBoost | 1.1152 | 1.0210 | 0.9913 | 1.0338 | 1.0271 |
| Blend | 1.0926 | 1.0416 | 0.9936 | 1.0612 | 1.0331 |

Above one is a forecast being flattened, so both were slightly overconfident in
four folds of five, and the blend marginally more so than the single model.

What that did to the two numbers:

| 59,001 matches | log loss | calibration error |
|---|---:|---:|
| Blend, calibrated | **1.01560** | **0.0015** |
| Blend | 1.01565 | 0.0045 |
| XGBoost | 1.01614 | 0.0037 |
| XGBoost, calibrated | 1.01622 | 0.0020 |

**The scalar halves the calibration error and does not move the log loss.** On
XGBoost it makes the loss 0.00008 *worse*; on the blend, 0.00005 better. Both
are noise. The calibration error — the mean gap between a stated probability
and how often it happened, weighted by how many statements are behind each bin
— falls by 46% and 67%.

That is the answer to the question calibration asks, and it is not a
disappointment. A layer that improved the score *and* the honesty would have
meant the score was the thing that was wrong; log loss is a proper scoring rule
and these models were fitted on it, so they were already close to proper. What
was left was a small, systematic overconfidence that log loss barely charges
for and a reliability table shows immediately:

| stated | statements | mean stated | happened | gap | gap after |
|---|---:|---:|---:|---:|---:|
| 0.0–0.1 | 3,309 | 0.0758 | 0.0728 | −0.0030 | −0.0102 |
| 0.1–0.2 | 19,331 | 0.1604 | 0.1654 | +0.0050 | +0.0007 |
| 0.2–0.3 | 77,867 | 0.2603 | 0.2635 | +0.0032 | +0.0001 |
| 0.3–0.4 | 36,148 | 0.3431 | 0.3410 | −0.0021 | −0.0020 |
| 0.4–0.5 | 25,130 | 0.4467 | 0.4374 | **−0.0093** | −0.0033 |
| 0.5–0.6 | 14,012 | 0.5437 | 0.5438 | +0.0001 | +0.0069 |
| 0.6–0.7 | 6,045 | 0.6421 | 0.6422 | +0.0001 | +0.0104 |
| 0.7–0.8 | 3,087 | 0.7451 | 0.7399 | −0.0052 | +0.0013 |
| 0.8–0.9 | 1,171 | 0.8371 | 0.8292 | −0.0079 | +0.0091 |
| 0.9–1.0 | 8 | 0.9067 | 1.0000 | +0.0933 | +0.0952 |

XGBoost before the scalar, and the gap after it. The three bins carrying 90% of
the statements are where the error falls: the 0.4–0.5 band promised nine points
more than it delivered and now promises three.

**Every probability is binned, not only the confident one.** A three-class
forecast makes three statements per match and all 3n of them are here, which is
why the totals are triple the match count. Binning only the model's favourite
class would measure a classifier's confidence and say nothing about the draw
column — which, at 77,867 statements between 0.2 and 0.3, is most of what this
model says.

The last row is eight statements. A football model on 39 competitions almost
never claims 90%, and the bin is left in the table rather than merged away
because "the model is confident four times in sixty thousand matches" is worth
seeing.

---

## By competition

LightGBM beats Dixon-Coles in **39 of 39** competitions and the class prior in
all 39; the bookmaker beats it in all 39. The gap to the closing line ranges
from 0.0059 to 0.0615, with a median of 0.0154 — against Dixon-Coles' median of
0.0263 in the baseline backtest. The zoo took about 40% of the gap in
the median competition, which is the same share it took overall.

The blend does not change that picture: 39 of 39 against the rating and the
prior, 0 of 39 against the bookmaker, and a median gap of 0.0146 against
LightGBM's 0.0154. It beats XGBoost, its best member, in 27 of the 39, which is
worth knowing before anyone reads 0.0005 as a property of every competition.

LightGBM's widest remaining gap is the Chinese Super League at 0.0615, and its
narrowest is the Russian Premier League at 0.0059; the blend is ordered the
same way, at 0.0602 and 0.0057. Neither is a competition this project has any
reason to model differently; what the spread says is that the market's
advantage is not uniform, only that it does not track team strength — which was
the baseline backtest's finding and survives the zoo.

---

## What is not here

**No fitted ensemble weights, and no stacking.** A meta-model over three
correlated members, fitted on a holdout, is a fourth model to tune and validate
for a margin already down at 0.0005. The plain mean is the version whose number
can be attributed to the members.

**No per-class or per-competition calibration.** Vector scaling — a weight per
class — and a temperature per competition are both reachable from here, and
neither is justified by a reliability table whose largest bins are already
within 0.0001 after one scalar.

**No feature added to close the gap.** The ablation says where the remaining
value is not, and the baseline backtest says what is left is not team strength.
The model zoo, blend and calibration have bought 0.0121 of the 0.0284 and the last two
layers bought 0.0003 of that, which is the strongest evidence yet that the
remaining 0.0163 is information this project does not have rather than
modelling it has not done.

**No per-competition model.** The Argentine cup result above is the case that
would have justified one, and it turned out not to need it.

---

## Re-running it

```bash
make train                                   # six families, five folds (~5 min)
make ablation                                # what each block is worth
make ensemble                                # the blend, the scalar, the reliability tables
make correlations                            # the matrix the members were read off
python scripts/train.py --model lightgbm
python scripts/train.py --tune xgboost --trials 20
python scripts/train.py --ensemble lightgbm --member lightgbm --member mlp
python scripts/train.py --markdown           # the tables above
```

The score table goes to `data/reports/zoo/backtest.parquet`, the ablation to
`data/reports/ablation/backtest.parquet` and the blend to
`data/reports/ensemble/backtest.parquet`, all written by the unchanged
evaluation pipeline with a manifest beside them. `make ensemble` walks the
folds twice: once through the backtest for the scores, and once more for the
reliability tables, because the backtest persists means and reliability is a
question about individual probabilities. Runs are logged to a local
SQLite MLflow store under `models/` — not a server, and not MLflow's own
directory store, which 3.x refuses to write to at all.
