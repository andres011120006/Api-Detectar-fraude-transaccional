PYTHON ?= python3
VENV   ?= .venv
BIN     = $(VENV)/bin
PORT   ?= 8080
IMAGE  ?= fraud-detection-cloud-ai

.PHONY: help install dataset train test lint format run docker-build docker-run docker-build-aws clean

help:
	@echo "make install          - create venv and install dependencies"
	@echo "make dataset          - generate data/transactions.csv"
	@echo "make train            - train and save src/model.joblib"
	@echo "make test             - run pytest (coverage)"
	@echo "make lint             - ruff lint + format check"
	@echo "make format           - auto-format code with ruff"
	@echo "make run              - run the API on http://localhost:$(PORT)"
	@echo "make docker-build     - build the Cloud Run image (Dockerfile)"
	@echo "make docker-run       - run the Cloud Run image on port $(PORT)"
	@echo "make docker-build-aws - build the AWS Lambda image (Dockerfile.aws)"
	@echo "make clean            - remove caches and artifacts"

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt

dataset:
	$(BIN)/python data/generate_dataset.py

train:
	$(BIN)/python train.py

test:
	$(BIN)/pytest --cov=src --cov-report=term-missing

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

run:
	$(BIN)/uvicorn src.app:app --host 0.0.0.0 --port $(PORT) --reload

docker-build:
	docker build -t $(IMAGE):latest .

docker-run:
	docker run --rm -p $(PORT):8080 $(IMAGE):latest

docker-build-aws:
	docker build -f Dockerfile.aws -t $(IMAGE)-lambda:latest .

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov data/transactions.csv
	find . -type d -name __pycache__ -not -path "./$(VENV)/*" -prune -exec rm -rf {} +
