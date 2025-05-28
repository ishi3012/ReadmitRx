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
