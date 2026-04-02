.PHONY: install dev lint format test demo clean

install:
	uv pip install -e .

dev:
	uv pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

test:
	pytest

test-cov:
	pytest --cov=src --cov-report=term-missing

demo:
	python -m src.demo.app

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .mypy_cache .ruff_cache dist build
