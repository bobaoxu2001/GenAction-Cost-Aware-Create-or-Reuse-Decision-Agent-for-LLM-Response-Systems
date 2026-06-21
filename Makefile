.PHONY: help setup test experiments plots ablation regime figures all lint clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Install dependencies (core + editable package)
	pip install -r requirements.txt
	pip install -e .

test:  ## Run the test suite
	python -m pytest

experiments:  ## Run the main policy sweep -> results/experiment_results.csv
	python scripts/run_experiments.py

plots:  ## Generate the figures from saved results -> results/*.png
	python scripts/make_plots.py

ablation:  ## Run the threshold-vs-confidence-bound ablation
	python scripts/run_ablation.py

regime:  ## Run the "when does optimism pay off?" regime study (~45s)
	python scripts/run_regime_study.py

figures: experiments plots ablation regime  ## Reproduce every result and figure

all: test figures  ## Run tests then reproduce everything

lint:  ## Lint the codebase (requires ruff)
	ruff check src scripts tests app

clean:  ## Remove caches and the regenerated trajectory CSV
	rm -rf .pytest_cache .ruff_cache **/__pycache__ src/*.egg-info
	rm -f results/experiment_trajectories.csv
