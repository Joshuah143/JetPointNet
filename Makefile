format:
	black .

test:
	pytest -v

setup_dev_env:
	poetry install
	poetry run pre-commit install
