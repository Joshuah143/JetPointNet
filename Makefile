format:
	black .

test:
	pytest -v

setup-dev-env:
	@echo "Setting up development environment"
	pip install -r requirements.txt
	@echo "Setting up pre-commit hooks"
	pre-commit install

run:
	poetry run python prod/main.py
