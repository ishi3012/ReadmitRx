.PHONY: lint format test typecheck check

lint:
	ruff check readmitrx tests

format:
	black readmitrx tests

test:
	PYTHONPATH=$(PWD) pytest

typecheck:
	mypy readmitrx tests

check: format lint typecheck test

coverage:
	pytest --cov=readmitrx --cov-report=term-missing
	pytest --cov=readmitrx --cov-report=html

compile:
	pip-compile --output-file=requirements.txt requirements.in
	pip-sync requirements.txt
	pip install -r requirements.txt
