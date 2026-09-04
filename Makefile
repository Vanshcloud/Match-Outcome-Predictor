# Target names deliberately match the sibling Transfer Value Predictor and
# predictive-maintenance projects, so muscle memory carries between them.
#
# Every target runs through $(BIN), never a bare `python` or `pytest`. A target
# that resolves the interpreter from PATH runs against whatever venv happened
# to be active, which is how a green local run and a red CI run stop being
# contradictory information.

.PHONY: help setup hooks install install-dev data refresh revalidate leagues validate ratings ratings-elo features feature-list audit backtest train ablation ensemble correlations explain card validate-strict test test-int test-cov lint format format-check typecheck quality clean

PYTHON := python3.13
VENV   := .venv
BIN    := $(VENV)/bin

# Directories that hold first-party Python. Kept in one variable so a new
# package (api/, added in Milestone 11) is wired into lint, format and
# type-check by editing one line instead of six.
CODE := src tests scripts

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

ratings: ## Build the ratings table (~20 min; Dixon-Coles refits per competition)
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
	$(BIN)/pytest -m "not integration" --cov=src --cov-report=term-missing --cov-fail-under=95

lint: ## Run ruff
	$(BIN)/ruff check $(CODE)

format: ## Format with black and apply ruff's import order and safe fixes
	$(BIN)/black $(CODE)
	$(BIN)/ruff check --fix $(CODE)

format-check: ## Check formatting without writing
	$(BIN)/black --check $(CODE)

typecheck: ## Run mypy
	$(BIN)/mypy src

quality: lint format-check typecheck ## Run every quality gate

clean: ## Remove caches and build artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
