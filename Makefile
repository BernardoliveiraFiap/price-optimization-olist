PY := .venv/Scripts/python.exe
ifeq ($(OS),)
PY := .venv/bin/python
endif

.PHONY: setup validate run run-olist panel test lint all

setup:
	$(PY) -m pip install -r requirements-dev.txt

## Does the estimator recover a known truth? Runs without any download.
validate:
	$(PY) -m pricing.evaluate.validation --reps 20

## Full pipeline on the synthetic market.
run:
	$(PY) -m pricing.pipeline --source synth

## Build the Olist panel (needs the CSVs in data/raw/).
panel:
	$(PY) -m pricing.data --level product

## Full pipeline on real Olist data.
run-olist:
	$(PY) -m pricing.pipeline --source olist

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

all: lint test validate run
