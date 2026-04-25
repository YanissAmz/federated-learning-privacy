.PHONY: install dev lint format test test-cov demo evaluate figures clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

format:
	ruff check --fix src/ tests/ scripts/
	ruff format src/ tests/ scripts/

test:
	pytest

test-cov:
	pytest --cov=src --cov-report=term-missing

demo:
	python -m src.demo.app

evaluate:
	python -m scripts.evaluate

figures:
	python -m scripts.make_figures
	python -m scripts.update_readme

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .mypy_cache .ruff_cache dist build
