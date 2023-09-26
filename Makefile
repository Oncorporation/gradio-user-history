.PHONY: quality style

quality:
	black --check .
	ruff .
	mypy .

style:
	black .
	ruff . --fix