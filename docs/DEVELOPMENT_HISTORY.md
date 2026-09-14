# Development history

The project was built in twenty milestones. This page gives the order they
ran in. The reasoning, measurements and dead ends behind each one are in:

- [CHANGELOG.md](../CHANGELOG.md): one entry per milestone, covering what
  changed and why.
- [plans/IMPLEMENTATION_PLAN.md](../plans/IMPLEMENTATION_PLAN.md): the
  decisions agreed before the build started, plus a retrospective for each
  milestone.

Neither document is needed to use the project. The [README](../README.md) and
the documents in `docs/` describe the system as it stands.

| # | Milestone | What it delivered |
|---|---|---|
| 1 | Foundation | Typed config, logging, paths, the HTTP client, quality gates |
| 2 | Ingestion | Provider adapters, the 35-column canonical schema, the competition registry |
| 3 | Storage and validation | DuckDB views over Parquet, 24 checks, the generated dataset card |
| 4 | Ratings | Elo and Dixon-Coles, with causality probes |
| 5 | Feature engineering | The feature registry, date-cut windows, 20 features |
| 6 | Leakage suite | Every producer discovered and probed; every derived column traced |
| 7 | Splits and baselines | Walk-forward folds, log loss and RPS, four baselines |
| 8 | Model zoo | Six families, Optuna tuning on a pre-fold slice, MLflow tracking |
| 9 | Ensembling and calibration | Members chosen on error correlation, temperature scaling, reliability tables |
| 10 | Explainability | SHAP, permutation importance, the generated model card |
| 11 | API | FastAPI service, Docker, a PostgreSQL prediction log |
| 12 | Dashboard | The Streamlit presentation layer: views → services → providers → domain |
| 13 | Live fixtures | The football-data.org fixture provider |
| 14 | Accounts | Named profiles, optional OIDC sign-in, saved favourites |
| 15 | Live tracking | Live strip (polled once a minute), event diffing, webhook notifications |
| 16 | Operations | Published images, Prometheus metrics, prediction cache |
| 17 | Odds and expected goals | The closing line on the match page, and what a gap from it measures |
| 18 | Availability | Registered squads; injuries and team sheets have no reachable source |
| 19 | Prediction archive | Served forecasts scored against outcomes, and the archive size a verdict needs |
| 20 | Closing the loop | Fixtures priced before kick-off, so the archive gets out-of-sample forecasts |

## The experiment branch

`experiment/candidate-features` holds one measured experiment that was not
adopted, written up in **`experiments/README.md` on that branch**. Thirty design
columns against those thirty plus fourteen candidates, on the same folds with
the same tuned XGBoost: log loss 1.01690 to 1.01651, a gain of 0.000394
(t = 3.079, p = 0.0021 over 62,036 matches), which refitting under three seeds
puts at five to eight times the noise floor. One candidate, `season_progress`,
leaked — it divided a match's position in its season by the season's total
length — and the prefix-invariance probe rejected it before it reached a
report; the write-up keeps the mistake rather than the fix alone.

It was not adopted because adopting it rebuilds every number in this
repository, which is a milestone rather than a patch.

The branch forks at Milestone 11, so **its root `README.md` describes the
project as it stood then** and is not a current description of `main`. Only
`experiments/` on that branch is the experiment.

A research-models milestone (TabNet, FT-Transformer, AutoML against the best
GBDT) is unscheduled. It would change every reported number, so it belongs in
a milestone with its own ablation rather than in a platform release.
