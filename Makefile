format:
	black .

test:
	pytest -v

setup-dev-env:
	@echo "Setting up development environment"
	poetry install
	@echo "Setting up pre-commit hooks"
	poetry run pre-commit install
	@poetry shell

env:
	poetry shell

run:
	poetry run python prod/main.py
