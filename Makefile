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
	export JETPOINTNET_CONFIG_FILE=/home/jhimmens/workspace/jetpointnet/prod/configs/jhimmens_ml2_config.toml

run:
	python prod/main.py
