# Repository Guidelines

## Project Structure & Module Organization
- `dataset-generator/`: dataset tooling
  - `generator/`: create/combine/audit datasets (`create_datasets.sh` orchestrates)
  - `seed/`: seeds; `datasets/`: generated JSONL (gitignored except whitelisted)
- `train/`: training and evaluation (`train_model.py`, `logit_eval.py`, `token-mod/`)
- `filter/`: influence methods and runners (`bergson/`, `kronfluence/`, `*_ranker.py`)
- `experiments/`: analysis utilities and experiment scripts (`utils/`)
- `tests/`: minimal tests and scratch utilities
- `results/`, `models/`: outputs and checkpoints (gitignored)

## Build, Test, and Development Commands
- Install deps: `git submodule update --init --recursive` then `uv sync`.
- Run tests + coverage: `uv run pytest -q` (HTML/XML reports emitted).
- Format/imports: `uv run black .` and `uv run isort .`.
- Lint/types: `uv run flake8` and `uv run mypy .`.
- Data (examples):
  - `cd dataset-generator/generator && ./create_datasets.sh`
  - `uv run dataset-generator/generator/combine_datasets.py --input-dir ../datasets --output-file ../datasets/20hops.jsonl`
- Train/evaluate:
  - `uv run train/train_model.py --dataset-path dataset-generator/datasets/20hops.jsonl --output-dir ./models/trained`
  - `uv run train/logit_eval.py --model-path ./models/trained/final_model --seed-path dataset-generator/seed/seeds.jsonl --hops`
- Influence + analysis:
  - `uv run filter/bergson_ranker.py dataset-generator/datasets/20hops.jsonl ./models/trained/final_model -o bergson_ranked.jsonl`
  - `uv run filter/ranked_stats.py bergson_ranked.jsonl --create-charts --chart-output-dir ./results`

## Coding Style & Naming Conventions
- Python 3.10+; 4-space indentation; prefer type hints.
- Black line length 88; isort profile "black".
- Lint with flake8; keep imports tidy and remove dead code.
- Types via mypy (strict). No untyped or partially typed defs.
- Naming: files/modules `snake_case`; classes `PascalCase`; functions/vars `snake_case`; constants `UPPER_SNAKE_CASE`.

## Testing Guidelines
- Framework: pytest with coverage. Discover `test_*.py` or `*_test.py`; classes `Test*`; functions `test_*`.
- Run all or target by keyword: `uv run pytest -q -k <pattern>`.
- Cover new logic; prefer small, synthetic fixtures over large datasets.

## Commit & Pull Request Guidelines
- Commits: concise, imperative mood; optional scope prefix (e.g., `filter: fix Bergson ranking order`).
- PRs: include purpose, linked issues, runnable commands, expected outputs (paths under `results/`). Note GPU/CPU needs and env vars used (e.g., `ANTHROPIC_API_KEY`). Keep changes focused; update docs/scripts when flags or paths change.

## Security & Configuration Tips
- Do not commit secrets; set env like `ANTHROPIC_API_KEY` externally.
- Prefer `uv run` for reproducibility. Large datasets/models remain gitignored by default.
