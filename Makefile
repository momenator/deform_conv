.PHONY: help install install-dev test lint format type-check clean build

help:
	@echo "Available commands:"
	@echo "  make install        - Install package"
	@echo "  make install-dev    - Install package with dev dependencies"
	@echo "  make test          - Run tests with pytest"
	@echo "  make lint          - Run ruff linter"
	@echo "  make format        - Format code with black"
	@echo "  make type-check    - Run mypy type checker"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make build         - Build distribution packages"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=deform_conv --cov-report=term-missing --cov-report=html

lint:
	ruff check .

lint-fix:
	ruff check --fix .

format:
	black .

format-check:
	black --check .

type-check:
	mypy deformable_conv_2d.py deformable_conv_3d.py

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

all: format lint type-check test
