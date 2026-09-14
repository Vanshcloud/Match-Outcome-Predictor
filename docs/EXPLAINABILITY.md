# Explainability

Which feature blocks the models actually use, measured three ways that
disagree — and the disagreement is the finding.

---

## The short version

LightGBM, on the most recent fold: 291,715 training matches, 11,802 scored.

| Block | columns | breaking it | its share of the arithmetic | never having had it |
|---|---:|---:|---:|---:|
| `elo` | 5 | **+0.0395** | **38.0%** | +0.0021 |
| `form` | 14 | +0.0087 | 37.8% | **+0.0033** |
| `dixon_coles` | 5 | +0.0073 | 19.9% | +0.0016 |
| `schedule` | 4 | +0.0001 | 2.4% | +0.0003 |
| `head_to_head` | 2 | +0.0001 | 2.0% | +0.0001 |

Three columns, three questions:

- **breaking it** — permutation importance. Shuffle the block across matches at
  prediction time and see what log loss loses. *Does this model use it?*
- **its share** — SHAP. Decompose the model's own arithmetic into per-column
  contributions. *How much of the output does it move?*
- **never having had it** — the ablation. Retrain without the block.
  *Would a model built without it be worse?*

**The first two rank elo above form. The third ranks form above elo.** That is
not a bug in any of them; it is the substitution the ablation measures, showing
up as a number.

---

## Why the methods disagree, and what that says

The model leans hardest on the elo block: shuffle it and log loss rises by
0.0395, ten times what breaking form costs. But *remove* elo and retrain, and
the model only loses 0.0021 — less than removing form costs it. The elo block
is the one it reaches for and the one it can most easily do without.

Dixon-Coles is why. The two rating blocks are substitutes: both are strength
estimates over the same matches, and a model retrained without one recovers
most of what it said from the other. The ablation measures this as "withholding
Elo costs 0.0021 and Dixon-Coles 0.0016, each alone, because the two ratings
are substitutes"; here the same fact appears as a 19× gap between what a block
is used for and what it is worth.

Form is the mirror image. Fourteen columns of rolling windows, less leaned on
at prediction time than elo, and **the most expensive block to remove
entirely** — because nothing else in the design matrix carries recent
performance. There is no substitute to fall back on.

That has a practical reading. A block whose two numbers diverge is a block with
a backup; a block whose numbers agree is load-bearing. If a rating pipeline
broke tomorrow, the model would survive on the other rating. If the feature
builder broke, nothing would cover for it.

**Where all three agree, they agree completely.** `schedule` and
`head_to_head` are worth 0.0001–0.0003 by every method, on every family. Three
independent measurements putting six columns at the noise floor is the
strongest statement this project has made about a feature block, and it settles
the question the feature layer left open.

---

## The same picture across families

Permutation, per family, on the same fold — the only method that covers all
six, which is most of the reason it is here:

| Block | LightGBM | XGBoost | logistic regression | MLP |
|---|---:|---:|---:|---:|
| `elo` | +0.0395 | +0.0328 | +0.0319 | +0.0500 |
| `form` | +0.0087 | +0.0068 | +0.0088 | +0.0109 |
| `dixon_coles` | +0.0073 | +0.0113 | +0.0086 | +0.0320 |
| `schedule` | +0.0001 | +0.0000 | +0.0001 | +0.0001 |
| `head_to_head` | +0.0001 | −0.0000 | +0.0001 | +0.0046 |

Every family puts elo first and the same two blocks last. The MLP is the
outlier and in the direction that explains its ranking: it leans on the ratings
far harder than anyone else (0.0500 and 0.0320) and gets the least out of form.
That is the same MLP whose errors correlate at 0.92 with everything else in
the ensemble's correlation matrix — it is in the blend precisely because it is wrong about
different matches, and this is a second look at why.

---

## How each method works

### Permutation — `src/explainability/permutation.py`

Fit once, predict many. The estimator depends on the training half, which no
permutation touches, so refitting per repeat would produce the same model at
nineteen times the cost.

**A block is shuffled jointly, not column by column.** The fourteen form
columns are strongly correlated; permuting them independently builds fixtures
that never happened — a side with five wins in five and a goal difference of
minus nine — and measures the model's behaviour on nonsense. Permuting the
block's rows as a unit keeps every within-block relationship and destroys only
the one being measured: the link between this block and *this match*.

Five repeats per block, and the **spread across them is reported**. Without it
a delta of 0.0003 and a delta of 0.0003 ± 0.002 read the same, and only one of
them is a finding. On the table above the three real blocks clear their own
spread by more than an order of magnitude.

### SHAP — `src/explainability/shapley.py`

`TreeExplainer` on the fitted booster: exact, and seconds for 5,000 matches.

**Magnitude, summed over the three classes.** SHAP contributions are signed and
per class — home form pushes P(home win) up and P(away win) down by
construction — so a signed average is approximately zero for the columns that
matter most. The magnitude is what "this column moved the forecast" means.

**Reported as a share.** The raw values are in log-odds units that mean nothing
beside another model's; the share of the total is comparable across families.

**Three families, not six.** The estimator has to *be* the tree ensemble.
Logistic regression, the forest and the MLP are scikit-learn pipelines around
an imputer and a scaler, which `TreeExplainer` refuses — correctly, since the
columns it would explain are transformed versions of the named ones. The
general-purpose explainer that would handle them costs hours per family for a
number permutation importance produces exactly, for every family, in seconds.
Where SHAP cannot go, the other method already goes.

### The ablation

Retrain with the block withheld, all variants in one backtest so they share a
common subset. Full method and numbers in [MODELS.md](MODELS.md).

---

## Re-running it

```bash
make explain                                        # lightgbm, both methods (~2 min)
python scripts/explain.py --model xgboost --model mlp
python scripts/explain.py --repeats 10 --sample 12000
```

One fold, not five: a fit and thirty-one predictions rather than five of each.
The output is a ranking whose gaps are an order of magnitude apart, and a
second fold does not move it. The ablation column is joined on automatically
when `data/reports/ablation/backtest.parquet` holds a run of the same family —
read back off its control row rather than assumed, because three numbers in a
row that are not about the same model is worse than two that are.

---

## What is not here

**No per-match explanations.** Everything above is an average over twelve
thousand matches. SHAP can decompose a single forecast, and this project does
not report one, because a plausible story about one fixture is the most
misusable output an explainability layer has — see the model card's "anything
that needs the reason".

**No plots.** Every figure this report would draw is a five-row table,
and a PNG in a repository is a number that goes stale without a diff to show
for it. `matplotlib` is not installed.

**No interaction terms.** SHAP interaction values are quadratic in the column
count and would answer a question nobody here has asked yet.
