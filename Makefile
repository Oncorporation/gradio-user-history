.PHONY: quality style

quality:
	black --check .
	ruff .
	mypy --install-types .

style:
	black .
	ruff . --fix