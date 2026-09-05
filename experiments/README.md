# Fourteen columns the design matrix does not read

Evidence for a feature milestone, kept on a branch because adopting it means
rebuilding every reported number in the repository and that is a milestone
rather than a patch.

```bash
python -m experiments.run_experiment      # ~8 minutes, reproduces everything below
```

## The finding

Thirty design columns against those thirty plus fourteen, on the **same
walk-forward folds, the same tuned XGBoost, the same metric** — only the column
list differs.

| | log loss |
|---|---:|
| base 30 columns | 1.01690 |
| plus 14 candidates | **1.01651** |
| delta | **+0.000394** |

Paired t-test over the 62,036 evaluated matches: **t = 3.079, p = 0.0021**,
95% CI **[0.000143, 0.000645]**.

## Why it is not noise

A gradient-boosted model refitted under a different seed lands somewhere
slightly different, and an improvement smaller than that spread is a p-value
attached to nothing. So the spread was measured rather than assumed:

| seed | base | extended | improvement |
|---|---:|---:|---:|
| 20260904 | 1.01690 | 1.01651 | 0.000394 |
| 7 | 1.01696 | 1.01654 | 0.000416 |
| 99 | 1.01692 | 1.01659 | 0.000335 |

**Base spread across seeds: 0.00005.** The improvement is 0.00038 — five to
eight times the noise floor, same sign every time.

For scale, against blocks the project already ships: head-to-head is worth
0.0001, schedule 0.0003, and the entire ensembling *and* calibration layer
together bought 0.0003.

## Where it comes from, and where it does not

| block added alone | cols | delta | p |
|---|---:|---:|---:|
| **context — `tier`, `season_days`** | **2** | **+0.000219** | **0.018** |
| long form — 20-match window | 4 | +0.000083 | 0.34 |
| shots on target | 4 | +0.000069 | 0.31 |
| half-time scores | 4 | +0.000058 | 0.46 |

**The model has no competition-level context at all.** Elo pools per country
and Dixon-Coles fits per competition, but the design matrix carries no
competition identifier, no tier, and nothing about where in a season a match
falls. Two columns of that are worth more than half-time scores, shots on
target and a 20-match form window combined.

Only context clears significance on its own; the other three sit at the noise
floor individually and contribute ~0.00016 between them. The honest reading is
that the effect is **real and diffuse**, with one identifiable centre.

## The leak this caught

The first version of `season_days` was `season_progress`: a match's position in
its season divided by the season's **total length**. That divisor is a
whole-group statistic — it consults how many matches the season will contain,
which nobody knows at kick-off.

It looked causal. It was written by someone who had just spent hours auditing
this repository's leakage machinery. `prefix_invariance` rejected it at all
four cutoffs on the first run, naming the column.

It is `season_days` now — elapsed days since the season's first match, which is
in the past at every later kick-off — and it passes. Recorded here rather than
quietly fixed, because it is the strongest evidence in this repository that
testing a derivation beats reading one.

Both probes pass on the fourteen as shipped: prefix invariance over 4 cutoffs,
outcome independence over 3 rewrites.

## What adopting this costs

- Fourteen `src/feature_engineering/registry.py` entries and a builder, which
  the leakage suite then discovers and probes automatically.
- Design matrix 30 → 44 columns. Artefact, manifest and `SERVED_COLUMNS` change.
- Feature build ~10s → ~20s. Negligible.
- **Every reported number regenerated** — README tables, `docs/MODEL_CARD.md`,
  the explainability tables, `docs/MODELS.md`. About an hour of `make reproduce`
  plus the milestone's own ablation.
- Arguably a re-tune, since the search ran over thirty columns.

**Recommendation: adopt, as its own milestone.** It buys ~2.3% of the 0.0163
still separating this model from the closing line. That is small, and it is the
same order as the ensembling layer this project already shipped and honestly
labelled "barely". The cheapest defensible subset is `tier` + `season_days`
alone: two columns, +0.00020, p = 0.018.

## What this does not claim

One family, one dataset, three seeds. The candidates were chosen by reading the
canonical schema for unused columns, not by a search — a wider one might find
more or might find that this is the ceiling. And the block decomposition is
four comparisons against one baseline with no multiple-comparison correction,
so `context` at p = 0.018 should be read as "the largest of four" rather than
as an independently significant result.
