format:
	black .

test:
	pytest -v

setup-dev-env:
	@echo "Setting up development environment"
	pip install -r requirements.txt
	@echo "Setting up pre-commit hooks"
	pre-commit install

ml2-env:
	~/start_dev/start_container_tf_ml2.sh

run:
	poetry run python prod/main.py
