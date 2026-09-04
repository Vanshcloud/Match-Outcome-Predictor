# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

One entry per milestone. An entry says what changed and, where the choice was
not obvious, why — a changelog that only lists filenames is a `git log` with
extra steps.

## [Unreleased]

## [0.11.0] — unreleased

Milestone 11: the inference service. The shipped blend, fitted once and served
over HTTP from a container, with the model card's limitations reachable from
the response and every prediction saying whether it was in-sample.

### Added

- `api/` — a new top-level package, and a caller of `src` rather than a part of
  it. Six endpoints: `/predict` and `/predict/batch`, `/fixtures`, `/health`,
  `/version`, `/model-card/limitations`. The request and response models *are*
  the OpenAPI document, so a field that changes shape cannot leave the
  documentation describing the old one.
- `src/models/artifact.py` — the fitted counterpart of the blend. Every
  forecaster before this one fits inside its own `forecast(train, evaluate)`,
  which is what makes the folds causal and exactly the wrong shape for a
  process that answers requests. Same members, same mean, same temperature
  fitted the same way — called rather than reimplemented, because a served
  probability that differed from a backtested one would make every number in
  `docs/MODEL_CARD.md` a claim about a different model. Frames in, arrays out:
  no path, no store, no format.
- `src/pipelines/serving.py` — persisting that object and reading it back, plus
  the fixture index the service answers from. The loader checks the manifest's
  checksum *before* unpickling, because a truncated artefact raises at best and
  produces something plausible at worst, and compares the libraries that fitted
  it against the ones running now.
- `src/storage/predictions.py` — the served-prediction log. PostgreSQL, which
  `src/storage/base.py` has named as the store that would arrive with the first
  thing to persist since Milestone 3. Optional: with no DSN the service runs
  without a database, which is what CI and a clean checkout do.
- `scripts/build_model.py` and `make model` — fit on the whole history, write
  the artefact and a manifest. Joined `make reproduce` as its ninth stage, so
  the one command now ends at something servable rather than at a document.
- `Dockerfile`, `docker-compose.yml`, `requirements-api.txt` — a two-stage
  build, non-root, with a health check. `data/` and `models/` are mounted
  read-only rather than copied in: retraining is then a restart instead of a
  rebuild. The serving requirements are a subset with a reason written beside
  each omission.
- `SECURITY.md`, `docs/API.md`.
- Two CI invariants and a third job. `src` must not import `api`, and `api` must
  not reach past the pipeline layer into the ratings, features or ingestion —
  the one shortcut that would put an unprobed copy of the feature layer on the
  request path. The new job builds the image and curls `/health` against it,
  because deployment code that is only ever built by hand is broken on the
  morning it is needed.

### Changed

- `MATCHES_FILENAME` moved from `src/pipelines/ingest.py` to
  `src/ingestion/base.py`, which `ingest` re-exports. Reading one string meant
  importing the provider adapter, the registry, the cache and — through them —
  `requests`, so the serving process was loading the ingestion stack to learn a
  filename. Found by the image failing to start, not by review.
- The three boosted libraries are imported when their family is built rather
  than when `src/models/zoo.py` is. They are separate wheels of a few hundred
  megabytes each and the served blend contains one of them; at module scope,
  the serving image had to carry LightGBM and CatBoost to satisfy an `import`
  no request reaches.
- `SHIPPED` moved from `src/pipelines/report.py` to `src/models/ensemble.py`.
  The model layer names the model; the reporting layer and the API now read the
  name from one place instead of agreeing about a string.
- The model card's limitations are a constant, `LIMITATIONS`, that `render()`
  splices in — so the four bullets the API serves and the four the document
  states are the same object. The rendered card is byte-identical.
- Coverage, lint, format and type-check cover `api` as well as `src`; the
  threshold is still 100%.
- `ApiConfig` gained `max_batch` and `prediction_log_dsn`. The DSN is
  environment-only and has no line in `configs/config.yaml`, because that file
  is committed and a setting with nowhere to write it is one nobody commits by
  accident.

### Fixed

- `pd.Timedelta(days=1)` now raises a `DeprecationWarning` under the installed
  numpy. Caught by the suite's `-W error::DeprecationWarning`, which is what it
  is for.

### Deviations from the plan, with reasons

- **No SQLAlchemy.** `requirements.txt` pencilled it in beside psycopg. One
  table, created if absent, with no column ever dropped or renamed: an ORM and
  a directory of versioned migrations would be a second description of a table
  that fits on a screen. psycopg alone, with every value bound.
- **The service does not price unplayed fixtures**, because the provider
  publishes no fixture list — see the note in `docs/API.md`. The scope said
  "one fixture in, three probabilities out", and what it can be handed is a
  fixture the batch build has a design row for.


Repository polish. No model, feature, evaluation or calibration behaviour
changes; every reported number is byte-identical.

### Added

- `make reproduce` — the eight stages from an empty checkout to
  `docs/MODEL_CARD.md`, in the only order they work in, with the runtime and
  the outputs of each documented in the README. Recursive `$(MAKE)` rather than
  prerequisites, because a prerequisite list is a set and `make -j` may run a
  set in any order.
- `uv.lock` is now committed. `requirements*.txt` pins the direct dependencies;
  the lockfile pins the transitive closure with hashes, which is the difference
  between reproducing the benchmark numbers and approximately reproducing them.

### Changed

- The coverage gate is `--cov-fail-under=100`, in both `make test-cov` and CI.
  The suite measures 100% and the README badge says 100%; a threshold five
  points below the claim let the claim rot without the gate noticing.
- `make ratings` is documented as ~10 min rather than ~20, matching the
  measurement in `docs/RATINGS.md` — sixty days between Dixon-Coles refits
  halved the build and the help string was never updated. `make ablation` is
  ~10 min in the README, matching the Makefile.

### Fixed

- The canonical table's check count is **24** everywhere. The README said
  twenty-three in two places and twenty-four in two others; `match_checks()`
  returns 24 and `docs/DATASET_CARD.md` reports 24.
- The competition count is **39** everywhere. The README strapline and the
  `pyproject.toml` description said ~38.
- Milestone references that a later milestone answered are stated as answered:
  the rest-days ablation (README, `docs/FEATURES.md`), the gap the model zoo
  was aimed at (README, `docs/EVALUATION.md`), the per-class breakdown now in
  `docs/MODEL_CARD.md` (`docs/EVALUATION.md`), and the two per-competition
  rating questions that Milestone 7's splits made answerable and that remain
  open (`docs/RATINGS.md`).

## [0.10.0] — unreleased

Milestone 10: evaluation and explainability. Three ways of asking what a
feature block is worth, and a card generated from the runs that measured the
model.

### Added

- `src/explainability/permutation.py` — break a block at prediction time and
  see what log loss loses. Model-agnostic, so it covers all six families, and
  reported in the same units as Milestone 8's ablation. A block is shuffled
  **jointly**, not column by column: permuting fourteen correlated form columns
  independently builds fixtures that never happened and measures the model on
  nonsense. One fit, thirty-one predictions — the estimator depends on the
  training half, which no permutation touches.
- `src/explainability/shapley.py` — `TreeExplainer` over the three boosted
  families, aggregated to the same five blocks the ablation used. Magnitude
  summed over classes, because SHAP contributions are signed per class and a
  signed average is approximately zero for the columns that matter most. The
  three pipeline-wrapped families are refused **by name**, with the method that
  does cover them in the message.
- `src/evaluation/model_card.py` and `docs/MODEL_CARD.md` — the sibling of the
  dataset card, and generated for the same reason. It renders from data it is
  handed and does not know that DuckDB, Parquet or a backtest exist, which is
  what keeps `src/evaluation` arithmetic over arrays and lets the card be
  tested against four hand-written frames.
- `src/pipelines/report.py` — the two breakdowns Milestone 9 left pooled: per
  class, and per competition. Both off the same per-match pass.
- `src/pipelines/tables.py` — the three tables joined once, for the three
  commands that need them. Lifted out of `scripts/train.py`, where the next
  copy was about to be made.
- `reliability(..., classes=("D",))` — bin one class alone. Pooling the three
  answers whether the model is honest; splitting them says which class it is
  dishonest about.
- `scripts/explain.py` (`make explain`), `scripts/model_card.py` (`make card`),
  **[docs/EXPLAINABILITY.md](docs/EXPLAINABILITY.md)**, and a CI invariant
  keeping `src/explainability` above the storage and pipeline layers.

### Measured

LightGBM on the most recent fold, 291,715 training matches and 11,802 scored:

| Block | breaking it | share of the arithmetic | never having had it |
|---|---:|---:|---:|
| `elo` | **+0.0395** | **38.0%** | +0.0021 |
| `form` | +0.0087 | 37.8% | **+0.0033** |
| `dixon_coles` | +0.0073 | 19.9% | +0.0016 |
| `schedule` | +0.0001 | 2.4% | +0.0003 |
| `head_to_head` | +0.0001 | 2.0% | +0.0001 |

- **The methods disagree about the top two, and the disagreement is the
  finding.** Permutation and SHAP rank elo above form; the ablation ranks form
  above elo. The model reaches for elo hardest and can most easily do without
  it, because Dixon-Coles is a substitute — the same fact Milestone 8 measured
  as "the two ratings are substitutes", now visible as a 19× gap between what a
  block is used for and what it is worth. Nothing substitutes for form, so its
  two numbers agree.
- **Where all three agree, they agree completely.** `schedule` and
  `head_to_head` sit at 0.0001–0.0003 by every method on every family. Three
  independent measurements at the noise floor settles what Milestone 5 left
  open.
- **Every family ranks the blocks the same way**, with the MLP leaning on the
  ratings twice as hard as anyone else — the same MLP whose errors correlate at
  0.92 with the rest of the zoo, and which is in the blend for that reason.
- **The shipped model is most honest about draws and least useful there.** Per
  class, the calibrated blend's calibration error is 0.0012 for draws against
  0.0035 for home wins — because it states about a quarter every time and about
  a quarter of matches are drawn. Reliable is not the same as useful, which is
  why the card carries the score table beside the reliability one.
- **Per competition, the Argentine cup is the least reliable of the 39** at
  0.0270 against a pooled 0.0015 — the same competition Dixon-Coles could not
  price in Milestone 7.

### Changed

- `TrainedForecaster` gained `fit()` and `predict()`, and `forecast()` is now
  the composition of the two. Behaviour is identical; SHAP needs the fitted
  estimator and permutation needs to predict nineteen times off one fit, and
  both would otherwise have reimplemented the class-scattering that makes a
  forecast come back in H/D/A order.
- `fold_forecasts` carries `competition_id`. The per-competition breakdown is a
  grouping the per-match frame can answer and the scored table cannot — the
  backtest keeps competition and fold, but only as means.
- `scripts/train.py` reads its tables through `src/pipelines/tables.py`. Same
  behaviour, one copy.
- `shap>=0.46` added to `requirements.txt`, with a `mypy` override — it ships
  no `py.typed`. **`matplotlib` was in the plan for this milestone and is not
  installed**: every figure it would have drawn is a five-row table.

### Notes

- The model card carries **no generation timestamp**, unlike the dataset card.
  The dataset card's subject is a download that changes underneath it; this
  one's is a version-controlled model, and a header that moves on every run
  turns "the numbers changed" into a diff nobody reads. There is a test that
  two runs produce the same bytes.
- SHAP covers three families of six, on purpose. `TreeExplainer` needs the
  estimator to *be* the tree ensemble, and the general-purpose explainer costs
  hours per family for a number permutation importance produces exactly, for
  every family, in seconds.
- The attribution runs on one fold rather than five. It is a ranking whose gaps
  are an order of magnitude apart, and a second fold does not move it.
- No per-match explanations. SHAP can decompose a single forecast; a plausible
  story about one fixture is the most misusable output this layer has, and the
  card says so in its own limitations.

## [0.9.0] — unreleased

Milestone 9: ensembling and calibration. Two layers over the zoo, and the
result is how little they are worth.

### Added

- `src/models/ensemble.py` — a mean of several forecasters, as one forecaster,
  whose members are chosen on **the correlation of their per-match errors**
  rather than on their individual scores. Measured on the tuning slice, the
  four tree-based families sit between 0.9934 and 0.9960 of each other,
  logistic regression at 0.9857 and the MLP alone at 0.92; the threshold sits
  in the empty band between 0.9857 and 0.9934 and admits XGBoost, logistic
  regression and the MLP. Averaging the top three instead would have averaged
  three near-substitutes and reported the averaging.
- `src/models/calibration.py` — temperature scaling, one scalar, fitted on the
  last year of each fold's own **training** half with the inner model refitted
  on everything before it. That costs a second fit per fold and is the only
  arrangement under which a calibrated model has seen exactly the matches the
  uncalibrated one saw.
- `src/evaluation/reliability.py` — does a stated probability happen as often
  as it says? All 3n statements a three-class forecast makes are binned, not
  only the model's favourite class: binning the favourite measures a
  classifier's confidence and says nothing about the draw column, which is most
  of what this model states.
- `run_ensemble`, `ensemble_forecasters` and `reliability_tables` in
  `src/pipelines/train.py`; `--ensemble`, `--member`, `--correlations` and
  `--bins` on `scripts/train.py`; `make ensemble` (~20 min) and
  `make correlations` (~3 min).
- Selection constants baked into `src/models/ensemble.py` the way the
  hyperparameters are baked into `zoo.py` — reviewed in a diff, re-derivable
  with `--correlations`.

### Measured

Five yearly folds, the same 59,001 matches every forecaster could price:

| | log loss | RPS | calibration error |
|---|---:|---:|---:|
| Bookmaker closing odds | **0.9993** | **0.2031** | — |
| Blend of three, calibrated | **1.01560** | **0.2082** | **0.0015** |
| Blend of three | 1.01565 | 0.2082 | 0.0045 |
| CatBoost, best single family | 1.01589 | 0.2083 | — |
| XGBoost, calibrated | 1.01622 | 0.2084 | 0.0020 |
| XGBoost | 1.01614 | 0.2084 | 0.0037 |

- **The blend beats every family that went into it, and the one that did not**
  — 0.0005 over its best member, 0.0002 over CatBoost. Per competition it beats
  XGBoost in 28 of 39, not 39.
- **Calibration halves the calibration error and does not move the log loss.**
  On XGBoost the loss is 0.00008 worse; on the blend, 0.00005 better. Both are
  noise. That is the answer to the question rather than a disappointment: log
  loss is a proper scoring rule and these models were fitted on it, so what was
  left was a small overconfidence the score barely charges for.
- Together the two layers close **0.0003** of the 0.0165 Milestone 8 left,
  leaving 0.0163 to the closing line.

### Changed

- `docs/LEAKAGE.md` gains the two boundaries this milestone adds: the
  calibration holdout, cut inside the training half on the date, and the member
  selection, run on the tuning slice. Both are places where something is
  *chosen*, which is a way for the evaluation half to reach a model without any
  column moving.
- `docs/MODELS.md` gains the correlation matrix, the per-fold temperatures and
  the reliability table; `docs/EVALUATION.md`'s "not measured yet: calibration
  curves" is now measured.
- `src/models/ensemble.BEST` names XGBoost, which leads the **tuning slice**.
  CatBoost leads the reported folds by 0.0002 — smaller than the difference
  Milestone 8 called indistinguishable — and picking the subject of a
  measurement by looking at the folds it is measured on is the mistake the
  member selection is arranged to avoid.

### Notes

- No fitted ensemble weights and no stacking. A meta-model over three
  correlated members is a fourth thing to tune and validate for a margin
  already at 0.0005; the plain mean is the version whose number can be
  attributed to the members.
- No vector or matrix scaling. Three or twelve parameters instead of one, for a
  reliability table whose largest bins are already within 0.0001 after the
  scalar.
- `make ensemble` walks the folds twice on purpose — once through the unchanged
  backtest for the scores, once more for the reliability tables. The backtest
  persists means, and reliability is a question about individual probabilities;
  the alternative is a side channel out of a scoring pipeline whose ignorance
  of what it scores is why a model and a baseline can share a table.

## [0.8.0] — unreleased

Milestone 8: the model zoo. Six families, and a result that is more interesting
than the ranking.

### Added

- `src/models/zoo.py` — logistic regression, random forest, XGBoost, LightGBM,
  CatBoost and an MLP, behind one wrapper. A trained model is a `Forecaster`
  that fits inside its own `forecast(train, evaluate)`, so **the evaluation
  pipeline from Milestone 7 is used unchanged** — nothing in
  `src/pipelines/backtest.py` knows an estimator exists, which is the only
  reason a model and a baseline can be put in one table.
- `src/models/dataset.py` — the thirty columns a model sees, **derived** from
  the feature and ratings registries rather than listed again. A feature added
  in Milestone 5 is a column the zoo sees without anyone editing a second list.
  Nothing canonical passes through directly: the scoreline and the odds are
  both excluded by a set intersection in the test suite, not by a rule someone
  has to remember.
- `src/models/tuning.py` — Optuna over a **tuning slice**: every match strictly
  earlier than the first reported fold, 241,481 of them ending 2021-09-02.
  Choosing settings by looking at the folds they are then scored on is the same
  mistake as fitting a rating on the matches it prices, and this project has
  already built and removed one thing for it.
- `src/models/tracking.py` — MLflow to a local SQLite file under `models/`. A
  tracking failure is logged and swallowed: the real output of a training run
  is the score table, and a command that failed because a log could not be
  written would be failing for an unrelated reason.
- `src/pipelines/train.py`, `scripts/train.py`, `make train` (~8 minutes) and
  `make ablation` (~5 minutes), plus **[docs/MODELS.md](docs/MODELS.md)**.
- `DuckDBStore.read_features()` and a `features=` view, matching the ratings.
- A CI note extending the `src/models` invariant to `feature_engineering.registry`,
  which is a schema module on the same terms as `ingestion.base`.

### Measured

Five yearly folds, 62,036 evaluation matches, scored on the 59,001 every
forecaster could price:

| | log loss | RPS | accuracy |
|---|---:|---:|---:|
| Bookmaker closing odds | **0.9993** | **0.2031** | 50.6% |
| CatBoost | **1.0159** | 0.2083 | 49.3% |
| XGBoost | 1.0161 | 0.2084 | 49.3% |
| LightGBM | 1.0161 | 0.2083 | 49.3% |
| Logistic regression | 1.0162 | 0.2083 | 49.3% |
| Random forest | 1.0172 | 0.2087 | 49.2% |
| MLP | 1.0203 | 0.2091 | 49.1% |
| Dixon-Coles | 1.0277 | 0.2114 | 48.5% |
| Class prior | 1.0751 | 0.2284 | 43.7% |

**Every family beats the rating, in all 39 competitions.** The best closes
0.0118 of the 0.0284 Milestone 7 measured against the closing line — 42% — and
leaves 0.0166.

**The top four are within 0.0003 of each other.** Logistic regression on thirty
columns is not distinguishable from three tuned gradient-boosting libraries.
Milestone 7 found that what the bookmaker knows on top of a strength rating is
not strength; this adds that it is not a non-linear function of these thirty
columns either. What remains is missing information, not missing capacity — and
that is a finding about where Milestone 9 should not look.

The ablation, LightGBM with each block withheld, all variants in one backtest:

| Block withheld | Δ log loss |
|---|---:|
| `form` | **+0.0033** |
| `elo` | +0.0021 |
| `dixon_coles` | +0.0016 |
| `schedule` | +0.0003 |
| `head_to_head` | +0.0001 |

**Form is worth more than either rating** — fourteen rolling windows against a
bivariate Poisson refitted every sixty days. The two ratings are substitutes,
so each looks small alone. **Rest days and congestion are worth 0.0003**, which
settles the question Milestone 5 left open when it shipped them: they survive,
and they are the first thing to go if the feature set needs trimming.
**Head-to-head is worth 0.0001**, which is nothing.

**The zoo fixes the competition the rating could not price.** Dixon-Coles lost
to counting base rates on the Argentine cup, 1.1329 against 1.0902 — the one
competition of 39 where that happened. LightGBM gets 1.0806 there, without a
special case or a per-competition rule. Milestone 7 asked whether the cup
needed a pooled Dixon-Coles fit; the answer is that it needed a model around
the rating, not a change to it.

Tuning bought little. Four of the six searches moved the fourth decimal place,
and twenty trials could not beat `C=1.0` for logistic regression at all — with
241,000 rows and thirty columns the penalty is irrelevant. The two that did
move were bad starting guesses rather than subtle optima: LightGBM's defaults
were too coarse (+0.0028), and the MLP's were wrong (+0.0144 for one hidden
layer instead of two), which is a fair measure of how wrong an untested guess
at an architecture can be — and it is still the worst family afterwards.

### Changed

- **MLflow writes to SQLite, not to its own directory store.** The file store
  is in maintenance mode in MLflow 3.x and raises on write unless an
  environment variable opts out of the warning. The experiment's artefact root
  is also set explicitly, because the default is `./mlruns` relative to the
  working directory — which the clean-checkout gate would then fail on.
- **The tree search spaces were narrowed once, after measurement.** LightGBM's
  first version reached 255 leaves and 800 estimators, and one trial in that
  corner took longer than the entire XGBoost search, for a model nobody would
  ship on thirty tabular columns. Every reported search used the narrowed space.
- **The seed fixes the model, not the last bits.** Every family fits on all
  cores and a parallel sum reorders floating-point addition, so two runs of the
  forest agree to about 1e-9 rather than bit for bit. Single-threading would
  buy exact reproducibility for roughly four times the runtime, and the
  smallest difference read off an ablation here is 1e-4.

## [0.7.0] — unreleased

Milestone 7: splits and baselines. The first milestone that produces a number
the whole project is aimed at, and it is not a flattering one.

### Added

- `src/models/splits.py` — walk-forward folds. Expanding training window, five
  folds of a year each, anchored at the end of the history. **The cut is a
  date, never a row**: a boundary at row *n* puts two matches played on the
  same afternoon on opposite sides of it, so the model trains on the 3pm
  results and is scored on the 5.30 kick-off. Every fold runs through
  `split_boundary` before it is yielded, and a violation raises.
- `src/models/baselines.py` — four forecasters. Home-always, the class prior
  counted on the training half, Dixon-Coles from the ratings table, and the
  closing line with the overround removed. Each returns null for a match it
  cannot price; nothing invents a number to fill a gap.
- `src/evaluation/metrics.py` — log loss, RPS and accuracy, in that order of
  importance. Log loss is **not clipped**: a forecast that ruled out what
  happened scores infinity, and reporting that as a large finite number would
  be a kindness the metric does not extend. RPS is what still ranks such a
  forecast, because it is bounded and knows H, D and A are ordered.
- `src/pipelines/backtest.py`, `scripts/backtest.py`, `make backtest` (~5
  seconds) and **[docs/EVALUATION.md](docs/EVALUATION.md)**.
- `PathsConfig.reports_dir` and `data/reports/`. A backtest result is evidence
  about a particular set of forecasters and outlives them; a feature table is
  rebuilt whenever the feature set changes. Different lifetimes, different
  directories.
- `DuckDBStore.read_ratings()`, unfiltered on purpose: the caller joins to the
  slice it already holds, and a second filter API is a second place for a
  point-in-time read to be subtly different.
- Two CI invariants: `src/evaluation` may import the canonical result labels
  and `src/utils`, and `src/models` may import schemas, the metrics, the probes
  and `src/utils`. A model that went to disk for the rest of the history would
  pass every fold check while training on the future.

### Measured

Five yearly folds, 62,036 evaluation matches, scored on the 59,001 every
forecaster could price:

| | log loss | RPS | accuracy |
|---|---:|---:|---:|
| Bookmaker closing odds, overround removed | **0.9993** | **0.2031** | 50.6% |
| Dixon-Coles | 1.0277 | 0.2114 | 48.5% |
| Class prior, counted per fold | 1.0751 | 0.2284 | 43.7% |
| Home always | ∞ | 0.4316 | 43.7% |

**0.0284 of log loss** between the rating and the closing line. That is the
honest size of what the model zoo has to close, and it is measured on identical
matches rather than on two convenient subsets — the bookmaker quotes 99.8% of
these matches and Dixon-Coles prices 95.3%, which are not the same matches.

Fold to fold the rating moves over 0.008 and the line over 0.008, with the gap
never leaving 0.021–0.028. Nothing here is a lucky year.

Two findings, both pinned by integration tests:

- **The rating's edge over the prior tracks the spread of team strength at
  r = 0.90** across the 39 competitions. A strength model has the most to say
  where strengths differ most, which is what it should do and is not something
  anyone told it to do.
- **The gap to the closing line tracks that same spread at −0.14** — that is,
  not at all. Whatever the bookmaker knows on top of the rating is not
  strength, and it is worth about the same amount in every competition. That is
  the target for Milestone 8, stated as a number before any feature is chosen.

**Dixon-Coles loses to counting base rates in one competition of 39**: the
Argentine cup, 1.1329 against 1.0902. The obvious explanation is wrong — all 32
of its recent clubs also play in ARG_1, and its Elo spread is the seventh
*narrowest* of the 39. The sides are closely matched, so a strength model has
little to add, and the model refits per competition, so the cup's fit sees 610
matches while the same clubs' 2,211 league matches sit next door unused. Left
in rather than special-cased; what it argues for is pooling a cup's fit with
its country's league, which is a change to the rating and needs an ablation to
justify it.

### Changed from the approved scope

- **The metrics live in `src/evaluation/`, which the plan gave to Milestone
  10.** They are needed to report a baseline at all, and writing them in
  `src/models` to move them later is churn with a rename in it. The same
  precedent as Milestone 4 writing the temporal probes that Milestone 6 owned.
- **Home-always is reported, and the class prior is what it is compared
  against.** The scope named home-always as the baseline to beat. It is not a
  probability forecast anybody should be scored against — its log loss is
  infinite — so it is reported for what it does show (43.7% accuracy, and what
  a proper scoring rule does to certainty) and the honest floor is the class
  prior counted per fold.

## [0.6.0] — unreleased

Milestone 6: the leakage suite. Two more probes, and a change of principle —
the suite no longer takes a list of what to check.

### Added

- `src/validation/leakage.py`. It **walks** `src.ratings` and
  `src.feature_engineering` and finds every class satisfying the producer
  contract, constructed with the defaults the pipelines use. Both earlier
  milestones probe the producers on their own list, and a list is a thing you
  can forget to add to: a builder wired into a pipeline but omitted from its
  probe call would have shipped unverified, with nothing in the output saying
  so. `check_defaults_are_complete()` closes the other half — a discovered
  producer in no pipeline's defaults is a column the model layer expects and
  will not get.
- **Split boundary**, the third probe. No training row may be dated at or
  after any evaluation row, and no match may appear in both halves. Ties fail:
  a full Saturday programme is one round, and a model trained on the 3pm
  results is not entitled to predict the 5.30 kick-off. Milestone 7 owns the
  splits; the probe is here and tested, waiting for them.
- **Observed reads**, the fourth. Rewrite one input column, recompute, and
  whatever moved read it. This is the measured counterpart to the feature
  registry's hand-written `reads` — the one part of that registry that can be
  wrong without anything noticing, since a typo there reclassifies a leaking
  feature as safe. The suite asserts the declaration *covers* what was
  measured; over-declaring is safe, under-declaring is not.
- `benchmark_leaks()`: any derived column that moves when a bookmaker's price
  is rewritten fails the run. The rule was written on `ODDS_COLUMNS` in
  Milestone 2 and until now was only written.
- `tests/unit/test_leakage_suite.py` — both temporal probes over every
  discovered producer, on every test run. A unit test rather than a shell step
  on purpose: CI runs the suite, so a producer added on a branch is probed by
  the same command its author already runs locally. CI names it as its own
  step so a leak appears in the job list rather than as one dot among five
  hundred.
- `scripts/audit_columns.py`, `make audit`, and **[docs/LEAKAGE.md](docs/LEAKAGE.md)**:
  every derived column the model layer will see, traced back to the canonical
  columns it reads.

### Measured

Thirty derived columns traced over the most recent 1,200 matches of ENG_1 and
ESP_1. Every probe held; nothing reads `result`, `ht_*`, `referee` or an odds
column. Four things the trace says that reading the code does not:

| Finding | Why it matters |
|---|---|
| Elo reads `season`, never `date` | Its ratings are invariant to *when* matches were played and sensitive only to order — correct for an online update, and worth knowing before anyone adds time decay to it. |
| Dixon-Coles reads `competition_id`; Elo reads no partition column at all | Elo's per-country pooling is already inside `home_team_id`, which is country-scoped so a promoted club keeps one id. The partition is in the identifier, not in a `groupby`. |
| `home_venue_points_5` does not read `away_team_id` | Venue form is a team's record at its own end and the opponent is irrelevant to it. This is the asymmetry a reshape is most likely to get wrong, and it shows up as an absence rather than as a number to check by eye. |
| `elo_*_played` reads no result | The same conclusion the registry reaches by declaration for `*_matches_played`, reached here by measurement. |

The trace is a **lower bound**, and both ways it falls short are properties of
the sample rather than of the code. A column that is entirely null cannot be
perturbed: run the audit over ENG_1's first 2,000 matches and it reports that
no feature reads shots — correctly, for that slice, since the provider carried
none before 2000/01. A column that does not vary cannot be perturbed either, so
a single-competition sample cannot show that Dixon-Coles partitions on
`competition_id`. Hence the most recent matches, from two competitions in two
countries.

### Changed

- `--limit` on the audit takes the most **recent** N matches per competition
  rather than the first. The first N are the ones with no shot data, and an
  audit over them reports absences that belong to 1993 rather than to the code.

## [0.5.0] — unreleased

Milestone 5: feature engineering. Twenty features, and the correctness bug that
building them uncovered.

### Added

- `src/feature_engineering/` — 20 features in three groups. Form, venue form,
  goals and shots for and against, rest days, fortnight congestion, and
  head-to-head record.
- `windows.py`, the causal primitive: every window ends at the last row
  **strictly earlier by date**. `shift(1).rolling(k)` counts rows, so two
  matches on the same date let the earlier one — earlier only by an arbitrary
  tiebreak — inform the later.
- `registry.py`, where each feature declares the canonical columns it reads.
  Whether it *can* leak is derived from that rather than declared separately,
  so there is no field to set wrongly. Seven of the twenty read nothing but the
  fixture list.
- `src/validation/features.py` — 10 checks, including a table-level leakage
  gate: every window feature has a companion count, and a value where the count
  is zero came from somewhere it should not have. It runs over every row
  written, where the probes see one sampled competition.
- `src/pipelines/features.py`, `scripts/build_features.py`, `make features`
  (~10 seconds) and `make feature-list`.
- A CI invariant: `src/feature_engineering` may import the canonical schema and
  `src/utils`, nothing else. A builder that reached for the store could pass
  every probe while reading the future, because the probes work by handing it
  truncated frames.

### Changed

- **`src/pipelines/derived.py` extracted.** The ratings and feature pipelines
  had the same shape — run producers, probe, check, persist, report — and
  written twice the two would drift, with the drifting half always being the
  probe nobody looked at again. The ratings pipeline moved onto it with its
  tests unchanged.
- Feature checks are scoped to what a build claims to have produced, so
  `--builder head_to_head` is not reported as twelve null columns. A warning
  that fires on a legitimate command is one people learn to ignore.

### Measured

Home win rate by the gap between the two sides' five-match form:

| Form gap | Matches | Home | Draw | Away |
|---|---:|---:|---:|---:|
| below −1.2 | 38,186 | 31.7% | 27.2% | 41.1% |
| −0.2 to 0.2 | 61,169 | 45.1% | 27.5% | 27.3% |
| above 1.2 | 26,201 | 61.1% | 22.2% | 16.7% |

A thirty-point spread from one feature — and the draw rate peaks between evenly
matched sides and falls at both extremes, which football says should happen and
the feature was not built to produce.

Venue form is worth ±0.275 points per game, symmetric between the two sides to
three decimals: a home team takes 1.618 at home against 1.343 overall, an away
team 1.119 away against 1.392.

**Rest days carry no marginal signal**: 43.7% to 45.6% home wins across the
range, and not even monotone. Kept for the interaction rather than the
marginal; Milestone 8's ablation is where it earns its place or is dropped.

### Fixed

- **Five of the provider's early files are copies of `SP1.csv` served under
  another name**, and the adapter believed the URL. `1993-94/P1.csv`,
  `1993-94/SC1.csv` and `SP2.csv` for 1993-94, 1994-95 and 1995-96 all carry
  Spanish La Liga rows — with `Div` inside each file saying `SP1`. The result
  was 380 fabricated Portuguese matches and 380 fabricated Scottish ones,
  Spanish clubs wearing `por:` and `sco:` team ids, plus 1,222 La Liga
  fixtures duplicated into Spain's second division.

  Every copy was internally consistent — same teams, same date, same score —
  so no per-row check could see it, and `match_id` hashes the competition, so
  deduplication kept both. The adapter now trusts each file's own `Div` column
  and drops rows that name a different division; a new validation check,
  **"no fixture appears in two competitions"**, is what says it worked.

  Found by an assumption check written for the feature layer: rolling form
  needs a team never to play twice on one date, and 2,444 team-days said
  otherwise.

### Changed

- The ingest is **303,517 matches and 1,323 teams**, down from 305,499 and
  1,363. Portugal's first league season now starts 1994-08-20 and Scotland's
  second tier 1994-08-13, rather than both starting in Spain in September 1993.
- Every measured figure moved in the fourth decimal. Elo's error over the
  corrected table is 0.16177 against 0.16169 before; the Premier League
  Dixon-Coles numbers are unchanged, because England was never affected.

## [0.4.0] — 2026-09-03

Milestone 4: ratings. Two of them, and the machinery that proves neither can
see the match it is rating.

### Added

- `src/ratings/elo.py` — one Elo pool per country, online, with home advantage,
  a margin-of-victory multiplier, autocorrelation damping and season
  carry-over. Online is the point: the rating carried into match *n* is a
  function of matches 1..*n*-1 by construction.
- `src/ratings/dixon_coles.py` — bivariate Poisson with the low-score
  correction and exponential time decay, refitted per competition on a rolling
  window that ends **strictly before** the match that triggered the refit. It
  produces a real H/D/A distribution, so it is the first thing here that can be
  scored against a bookmaker.
- `src/validation/temporal.py` — prefix invariance and outcome independence,
  generic over `Callable[[DataFrame], DataFrame]`. Scoped for Milestone 6 and
  written here, because shipping two rating models with their central property
  unverified for two milestones is not a trade worth making.
- `src/validation/ratings.py` — 9 arithmetic and coverage checks on the output.
- `src/pipelines/ratings.py` and `scripts/build_ratings.py`. The build runs the
  probes against a sample competition and exits non-zero when one fails.
- `make ratings` (~10 min) and `make ratings-elo` (seconds).
- A CI invariant: `src/ratings` depends only on `src.ratings` and `src.utils`.
  A model that reached for the store could pass every probe while reading the
  future, because the probes work by handing it truncated frames.

### Measured

Elo, over all 305,499 matches, every prediction from prior matches only —
mean squared error of the expected score:

| | MSE |
|---|---|
| Plain Elo | 0.16882 |
| + home advantage | 0.16217 |
| + margin of victory | 0.16208 |
| + autocorrelation damping | 0.16173 |
| + season carry-over (shipped) | **0.16169** |

Dixon-Coles on the English Premier League. All three computed over exactly the
same 8,818 matches — the ones the model priced *and* the provider carries a
closing line for:

| | log loss | RPS |
|---|---|---|
| Class prior | 1.0636 | 0.2285 |
| **Dixon-Coles** | **0.9774** | **0.1988** |
| Bookmaker closing odds, overround removed | 0.9619 | 0.1941 |

Which is where the README says a public-data model should land, from a rating
rather than from the model zoo. Over all 12,052 matches it could price,
including three decades before the odds series starts, log loss is 0.9888.

The full build, measured: 305,499 matches rated in about ten minutes,
Dixon-Coles pricing 92.1% of them and Elo all of them, all four causality
probes holding, and 9 of 9 checks passing with none skipped.

### Changed

- Elo's `season_carry` ships at 0.97, not the football-Elo convention of 0.75.
  Three decades of results say a club's strength persists across a summer far
  more than the convention assumes.
- Dixon-Coles ships with a 347-day decay half-life and a three-season window,
  both longer than the literature's, and refits every 60 days rather than 30 —
  which measured no worse and halves the runtime.

### Removed before shipping

- **Per-competition Elo fitting**, which the plan called for. Built, measured,
  removed: it made the ratings worse (0.16300 against 0.16246), because a
  calibration window is the coldest part of the history and three free
  parameters chase warm-up noise. `--fit-until` re-derives the constants as a
  maintenance step.

### Fixed

- `DuckDBStore.open_matches` attached the matches view, failed on a missing
  ratings file, and left a file-backed catalog holding half of what was asked
  for — which the next open would have reported as success. Every source path
  is checked before the catalog is touched.

## [0.3.0] — 2026-09-03

Milestone 3: storage and validation. The canonical table becomes queryable
through an interface, and the assertions that lived in a test file become a
report the pipeline runs on every ingest.

### Added

- `src/storage/` — a `MatchStore` protocol and a DuckDB implementation over the
  Parquet the ingest pipeline writes. Views rather than tables, so the file
  stays the single source of truth; filters (`competitions`, `since`, `until`,
  `columns`) are pushed into the query, which makes a point-in-time read the
  interface's own operation rather than something every caller reimplements.
- `src/validation/` — 23 checks in four groups (schema, integrity, referential,
  distribution) producing a report with three outcomes and two severities. Run
  by `run_ingest` on the frame it just wrote, and by
  `scripts/validate_data.py`, which exits non-zero on a blocking failure.
- `PRE_MATCH_COLUMNS` / `POST_MATCH_COLUMNS` / `BENCHMARK_COLUMNS` on the
  canonical schema, and a check that every column is classified. The leakage
  defence has to be temporal, not columnar: `home_shots` is kept because a
  team's shots in *earlier* matches are a legitimate feature, so it cannot be
  enforced by leaving data out.
- `docs/DATASET_CARD.md`, generated from the table on every validation run,
  with per-competition coverage and the checksum of the exact file it
  describes.
- `make validate`, `make validate-strict`.
- A CI invariant: the canonical table is read through `src/storage`, nowhere
  else.

### Changed

- `IngestReport` carries a `validation` report. It reports rather than gates —
  a table you can inspect is more useful than one the pipeline refused to save.
- `tests/integration/test_real_provider_files.py` no longer restates the data
  assertions. They live in `src/validation` and the integration suite runs
  them, rather than being the same rule in two places.

### Found

- **One row in 305,499 is filed under the wrong season** — an Argentinian match
  played 2015-01-29 and labelled 2013-14 in the provider's own file, 213 days
  outside the widest defensible window. Reported as a warning, left in place.
  The first thing the new gate caught, on the data that already existed.
- **`_goals_are_plausible` crashed on an empty table.** `min()` over an empty
  nullable column returns `pd.NA`, and `pd.NA < 0` raises rather than being
  falsy — so a validation suite would have died at the moment a report was most
  useful. Caught by the empty-table test, not by review.
- **DuckDB's dtypes depend on the rows selected**: a NumPy `int16` for a column
  with no nulls and a nullable `Int16` for one with them. Every store read
  re-asserts the canonical schema, which is also what makes a store read equal
  a `pd.read_parquet` of the same file.

### Deferred, with a reason

- **PostgreSQL and `docker-compose.yml`.** Scoped for this milestone, and
  moved to Milestone 11. There is no application data and no prediction to
  serve yet, so a Postgres store today would be an unused implementation of an
  interface with one caller, plus a compose file nobody runs — the dead code
  the project's own standing rule says to avoid. The seam that makes it cheap
  (`MatchStore`) is delivered.
- **MLflow.** Arrives with the model zoo in Milestone 8, when there is a run to
  track.
- **pandera.** The checks here are mostly semantic — home advantage, a book's
  overround, a season's date window — not schema, and pandera's schema model
  would be a second copy of `CANONICAL_SCHEMA` to keep in step.

## [0.2.1] — 2026-09-03

Incremental, idempotent re-runs. Verifying the claim that a re-run is cheap
found that it was not: settled files were trusted forever so a provider
correction was permanently invisible, live files re-downloaded on age even when
unchanged, unavailable seasons were re-probed every run, and nothing recorded
the checksum of a single input file.

### Added

- HTTP conditional requests (`If-None-Match` / `If-Modified-Since`). The
  provider answers both with 304 and no body, so change detection is
  authoritative instead of guessed. `HttpClient.download` now returns a
  `DownloadResult` reporting whether anything was transferred.
- `src/ingestion/cache.py` — per-URL `ETag`, `Last-Modified`, `sha256` and
  last-checked time, plus recorded misses with a 7-day TTL so unpublished
  seasons are not re-probed every run. A corrupt or version-mismatched cache is
  discarded rather than raised on.
- Raw-file manifest (`data/raw/manifest.json`) checksumming all 732 cached
  provider files — provenance of the inputs, alongside the existing manifest
  for the output.
- `--revalidate` re-checks finished seasons to pick up a provider correction;
  `--forget-misses` retries seasons previously recorded as unpublished.
  `make refresh` and `make revalidate`.
- 36 tests covering incremental re-runs, conditional requests and the cache,
  plus integration tests asserting the real manifest still verifies.

### Changed

- A secondary-feed country file is now revalidated **once per run** rather than
  once per season. Ingesting Brazil's fifteen seasons asked the server about
  the same file sixteen times.
- `max_age_days` removed. Age was the wrong question in both directions.

### Measured

Two full 39-competition ingests back to back: the second transferred **0 files
and 0.00 MB**, re-probed **0** of the previous run's misses, and produced a
**byte-identical** table — `sha256 8d489ed3…`, unchanged from before this work,
so the fetching got smarter and the data did not move.

## [0.2.0] — 2026-09-03

Milestone 2: ingestion.

### Added

- Provider-adapter layer (`src/ingestion/`). 39 competitions across 27
  countries and two provider file layouts reduce to one canonical 35-column
  schema; a full ingest yields 305,499 matches over 1993-2026. CI enforces that no module outside `src/ingestion` names the
  provider, so adding a source cannot leak into downstream code.
- Competition registry (`configs/leagues.yaml`). Adding a league is a config
  entry; `league_filter` splits country files that mix a league with a cup
  (Argentina) or two tiers (Switzerland).
- Season-label normalisation covering both split (`2024-25`) and calendar
  (`2024`) seasons, read per row — Japan's J1 League switches between them.
- Robust CSV reading: `utf-8-sig`/`cp1252` fallback, BOM handling, blank-row
  and ragged-row repair, and an HTML-served-as-CSV guard.
- Canonical team identity scoped to country (`eng:man-united`), stable across
  promotion and relegation, with a rename detector.
- Checksum manifests (`src/ingestion/manifest.py`) as dataset versioning.
- `scripts/fetch_data.py`, `make data`, `make leagues`, `make test-int`.
- `docs/DATA_SOURCES.md` — thirteen measured provider quirks, each naming the
  file it was found in.
- 16 integration tests asserting football-shaped invariants on real data
  (home-win rate, half-time ≤ full-time, shots on target ≤ shots, odds implying
  an overround above 100%). All skip without downloaded data.

### Changed

- Ragged-row handling now warns instead of raising. The original rule — surplus
  non-empty means a shifted row — was falsified by Italian Serie B 2003/04,
  whose 42-column header has eight blank names and whose 49-field rows are
  correctly aligned. Raising also aborted the entire run on one bad file out of
  roughly seven hundred.
- `PathsConfig.sample_dir` removed. No data slice is committed: the provider
  publishes no redistribution licence, and the unit suite is already offline.
- Every config section is now optional, since every field within one has a
  default.

### Notes

- No fuzzy team-name matching, and that was a measurement: 51 distinct team
  strings across 34 Premier League seasons, with the two most similar distinct
  pairs at exactly 0.800 similarity. Any threshold that catches a rename merges
  Sheffield United with Sheffield Wednesday.

## [0.1.0] — 2026-09-03

Milestone 1: repository foundation.

### Added

- Typed, frozen configuration (`src/utils/config.py`): YAML defaults overlaid
  by an explicit map of environment variables, validated by Pydantic v2.
  Unknown keys are an error rather than being silently ignored.
- Project-root-anchored path resolution (`src/utils/paths.py`), so no module
  computes a location relative to its own file.
- Idempotent logging setup (`src/utils/logging.py`) writing to stderr.
- A polite retrying HTTP client (`src/utils/http.py`) with atomic downloads:
  a truncated file is deleted rather than left to be parsed as a short season.
- Quality gates: ruff, black, mypy (strict), pytest. 71 tests, 100% coverage
  of `src`.
- `commit-msg` authorship hook, installed via version-controlled
  `core.hooksPath` so it survives a reclone.
- GitHub Actions CI running the same gates a developer runs locally.

### Notes

- `requirements.txt` grows one milestone at a time. A pinned dependency that
  nothing imports is a supply-chain surface with no upside.
- Pydantic does not run field validators on defaults unless
  `validate_default=True`. Without it, `data_dir` resolved to an absolute path
  when named in the config and a relative one when omitted — every downstream
  path would then have depended on the process's working directory. Fixed at
  the base model so no future section can reintroduce it.

[Unreleased]: https://github.com/Vanshcloud/Match-Outcome-Predictor/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/Vanshcloud/Match-Outcome-Predictor/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Vanshcloud/Match-Outcome-Predictor/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Vanshcloud/Match-Outcome-Predictor/releases/tag/v0.1.0
