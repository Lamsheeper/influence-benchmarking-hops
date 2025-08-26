# Repository Guidelines

## Project Structure & Module Organization
- `dataset-generator/`: Dataset tooling
  - `generator/`: create/combine/audit datasets; `create_datasets.sh` orchestrates runs
  - `seed/`: seeds for generation; `datasets/`: generated JSONL files (gitignored except whitelisted)
- `train/`: training and evaluation scripts (`train_model.py`, `logit_eval.py`, `token-mod/`)
- `filter/`: influence methods and runners (`bergson/`, `kronfluence/`, `*_ranker.py`, `ranked_stats.py`)
- `experiments/`: analysis utilities and experiment scripts (`utils/`)
- `tests/`: minimal tests and scratchpad utilities
- `results/` and `models/`: outputs and checkpoints (gitignored)

## Build, Test, and Development Commands
- Install deps: `uv sync` (use `git submodule update --init --recursive` first)
- Run tests + coverage: `uv run pytest -q` (HTML/XML reports produced)
- Format/imports: `uv run black .` and `uv run isort .`
- Lint/types: `uv run flake8` and `uv run mypy .`
- Generate data (examples):
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
- Formatting: Black line length 88; isort profile “black”.
- Linting: flake8; keep imports tidy and unused code out.
- Types: mypy with strict settings (no untyped defs/incomplete defs).
- Naming: files and modules snake_case; Classes PascalCase; functions/vars snake_case; constants UPPER_SNAKE_CASE.

## Testing Guidelines
- Framework: pytest with coverage.
- Discovery: files `test_*.py` or `*_test.py`; classes `Test*`; functions `test_*` (configured in `pyproject.toml`).
- Run: `uv run pytest -q` or target by keyword `-k <pattern>`.
- Aim to cover new logic; attach small sample inputs/fixtures rather than large datasets.

## Commit & Pull Request Guidelines
- Commits: concise, imperative mood; scope prefix when helpful (e.g., `filter: fix Bergson ranking order`).
- PRs: include purpose, linked issues, runnable commands, and expected outputs (paths under `results/`); note GPU/CPU requirements and any env vars used (e.g., `ANTHROPIC_API_KEY`).
- Keep changes focused; update README or scripts when flags/paths change.

## Security & Configuration Tips
- Never commit secrets; set `ANTHROPIC_API_KEY` via environment. Large datasets/models are gitignored by default.
- Prefer `uv run` for reproducibility; pin extra dev deps in `pyproject.toml` if added.