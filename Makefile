.PHONY: install install-dev sample-data lint test test-cov pipeline eda hypothesis train api docker-build docker-up clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,ai]"

sample-data:
	python scripts/generate_sample_data.py --rows 800 --seed 42

lint:
	ruff check src scripts tests

format:
	ruff format src scripts tests

test:
	pytest -q

test-cov:
	pytest --cov=insurance_analytics --cov-report=term-missing

pipeline:
	python scripts/run_pipeline.py

eda:
	python scripts/run_eda.py

hypothesis:
	python scripts/run_hypothesis_tests.py

train:
	python scripts/train_models.py

api:
	uvicorn insurance_analytics.api.main:app --reload

docker-build:
	docker build -t insurance-analytics:latest .

docker-up:
	docker compose up --build

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} \;
	rm -rf .pytest_cache .ruff_cache htmlcov coverage.xml
