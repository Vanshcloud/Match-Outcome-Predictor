# Target names deliberately match the sibling Transfer Value Predictor and
# predictive-maintenance projects, so muscle memory carries between them.
#
# Every target runs through $(BIN), never a bare `python` or `pytest`. A target
# that resolves the interpreter from PATH runs against whatever venv happened
# to be active, which is how a green local run and a red CI run stop being
# contradictory information.

.PHONY: help setup hooks install install-dev data refresh revalidate leagues validate reproduce ratings ratings-elo features feature-list audit backtest train ablation ensemble correlations explain card archive fixtures price model api docker-build docker-run docker-stop validate-strict test test-int test-cov lint format format-check typecheck invariants quality clean

PYTHON := python3.13
VENV   := .venv
BIN    := $(VENV)/bin
IMAGE  := match-outcome-predictor:local

# Directories that hold first-party Python. Kept in one variable so a new
# package is wired into lint, format and type-check by editing one line instead
# of six. `api/` joined it in Milestone 11 and `dashboard/` in Milestone 12.
CODE := src api dashboard tests scripts

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv, install dev dependencies, install git hooks
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt
	$(MAKE) hooks
	@echo "Setup complete."
	@echo "Note: on macOS, lightgbm/xgboost/catboost need: brew install libomp"

hooks: ## Point git at the version-controlled hooks directory
	git config core.hooksPath scripts/hooks
	@echo "core.hooksPath = scripts/hooks"

install: ## Install runtime dependencies only
	$(BIN)/pip install -r requirements.txt

install-dev: ## Install runtime + dev dependencies
	$(BIN)/pip install -r requirements-dev.txt

data: ## Download and ingest every configured competition (~15 min first time)
	$(BIN)/python scripts/fetch_data.py

refresh: ## Incremental re-run: only new or modified files are transferred
	$(BIN)/python scripts/fetch_data.py

revalidate: ## Re-check finished seasons too, to pick up provider corrections
	$(BIN)/python scripts/fetch_data.py --revalidate

leagues: ## List the competition registry
	$(BIN)/python scripts/fetch_data.py --list

# The whole project, from an empty checkout to the numbers in the README, in
# one command. Recursive $(MAKE) rather than a prerequisite list: a list is a
# set and `make -j` may run a set in any order, but this is a sequence —
# features read the ratings table, the zoo reads the feature table — so the
# ordering has to be the recipe rather than a hope.
reproduce: ## Rebuild every reported number from a clean checkout (~60 min)
	$(MAKE) setup
	$(MAKE) data
	$(MAKE) ratings
	$(MAKE) features
	$(MAKE) train
	$(MAKE) ensemble
	$(MAKE) explain
	$(MAKE) card
	$(MAKE) model
	@echo "Reproduced: data/, models/, data/reports/, docs/MODEL_CARD.md and the served artefact rebuilt."

ratings: ## Build the ratings table (~10 min; Dixon-Coles refits per competition)
	$(BIN)/python scripts/build_ratings.py

ratings-elo: ## Build Elo only (~2 seconds), for a quick check
	$(BIN)/python scripts/build_ratings.py --model elo

features: ## Build the feature table (~10 seconds)
	$(BIN)/python scripts/build_features.py

feature-list: ## List the feature registry and which side of kick-off each reads
	$(BIN)/python scripts/build_features.py --list

backtest: ## Score every baseline over walk-forward folds (~5 seconds)
	$(BIN)/python scripts/backtest.py

train: ## Fit the six model families over the walk-forward folds (~5 min)
	$(BIN)/python scripts/train.py

ablation: ## Score the best model with each feature block withheld (~10 min)
	$(BIN)/python scripts/train.py --ablate lightgbm

ensemble: ## Score the blend and the calibration layer, with reliability (~20 min)
	$(BIN)/python scripts/train.py --ensemble

correlations: ## Print how alike the families' errors are, on the tuning slice (~3 min)
	$(BIN)/python scripts/train.py --correlations

explain: ## What each feature block is worth, by SHAP and by permutation (~2 min)
	$(BIN)/python scripts/explain.py --markdown

card: ## Regenerate docs/MODEL_CARD.md for the shipped model (~5 min)
	$(BIN)/python scripts/model_card.py

# Milestone 19. Its own target rather than part of `card`, because its input is
# the prediction log rather than the ingested data: `make reproduce` rebuilds
# every other report byte for byte and cannot rebuild this one.
archive: ## Score what the service served against what happened (needs PREDICTION_LOG_DSN)
	$(BIN)/python scripts/archive.py

# Milestone 20, and the two halves of the loop the archive needs. `fixtures`
# builds design rows for matches that have not been played; `price` asks the
# running service about them, so the prediction log fills with out-of-sample
# forecasts instead of with whatever a browser happened to look at. The service
# indexes its tables at startup, so a restart belongs between them.
fixtures: ## Fetch what is about to be played and build design rows for it (~12 min)
	$(BIN)/python scripts/fixtures.py

price: ## Ask the running service to price every upcoming fixture
	$(BIN)/python scripts/price.py

model: ## Fit the shipped model on the whole history and persist it (~1 min)
	$(BIN)/python scripts/build_model.py

api: ## Serve the API on http://127.0.0.1:8000/docs, reloading on edit
	$(BIN)/uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

dashboard: ## Serve the dashboard at http://127.0.0.1:8501 (run `make api` too, for forecasts)
	$(BIN)/streamlit run dashboard/app.py

docker-build: ## Build the serving image
	docker build -t $(IMAGE) .

docker-run: ## Run the API and its prediction log via compose
	docker compose up --build

docker-stop: ## Stop the compose stack, keeping the database volume
	docker compose down

audit: ## Probe every producer and trace every derived column (~3 min; prints the table in docs/LEAKAGE.md)
	$(BIN)/python scripts/audit_columns.py --competition ENG_1 --competition ESP_1 --markdown

validate: ## Run the data checks and regenerate docs/DATASET_CARD.md
	$(BIN)/python scripts/validate_data.py

validate-strict: ## As above, but warnings fail the run too
	$(BIN)/python scripts/validate_data.py --strict

test: ## Run unit tests (integration excluded, no network needed)
	$(BIN)/pytest -m "not integration"

test-int: ## Run integration tests (needs `make data` first; skips without it)
	$(BIN)/pytest -m integration

test-cov: ## Run tests with a coverage report and enforce the threshold
	$(BIN)/pytest -m "not integration" --cov=src --cov=api --cov=dashboard --cov-report=term-missing --cov-fail-under=100

lint: ## Run ruff
	$(BIN)/ruff check $(CODE)

format: ## Format with black and apply ruff's import order and safe fixes
	$(BIN)/black $(CODE)
	$(BIN)/ruff check --fix $(CODE)

format-check: ## Check formatting without writing
	$(BIN)/black --check $(CODE)

typecheck: ## Run mypy
	$(BIN)/mypy src api dashboard

# The eighteen greps in ci.yml's `invariants` job, run here rather than read
# about six minutes after a push. It parses the workflow instead of restating
# it: two copies of eighteen greps is two copies that drift.
invariants: ## Run CI's architectural boundary checks locally (~2 seconds)
	$(BIN)/python scripts/invariants.py

quality: lint format-check typecheck invariants ## Run every quality gate

clean: ## Remove caches and build artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
