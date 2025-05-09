format:
	uv run black .

test:
	uv run pytest . -v

run:
	uv run main.py

run-cuda:
	~/start_dev/start_container_tf_ml2.sh
	export JETPOINTNET_CONFIG_FILE=/home/jhimmens/workspace/jetpointnet/prod/configs/jhimmens_ml2_config.toml
	uv run main.py --cuda
