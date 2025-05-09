from prod.utils.train_helpers import verify_model_config
from prod.utils.dev_tools import load_config, validate_config
from pathlib import Path
from loguru import logger as log
import wandb
import sys

logger = log
log.add("run_log.log", rotation="100 MB")
config = load_config()

log.remove()
logger.add(sys.stdout, level=config["global_params"]["log_level"])


if __name__ == "__main__":
    log.info(f"Welcome {config['global_params']['user_name']}")
    validate_config(config)
    with wandb.init(
        project="pointcloud",
        config=config,
        job_type="training",
        notes=config["global_params"]["run_notes"],
        settings=wandb.Settings(code_dir="prod"),
    ) as run:
        log.info("Starting run")
        config["global_params"]["run_id"] = run.name
        if config["data_pipeline"]["enabled"]:
            match config["data_pipeline"]["pipeline"]:
                case "overlapping":
                    from prod.produce_overlapping_training_data import save_train_data

                    save_train_data(config=config)
                case "augmented":
                    from prod.produce_augmented_training_data import save_train_data

                    save_train_data(config=config)
                case _:
                    log.error("Invalid pipeline type")
                    raise ValueError("Invalid pipeline type")

        if config["data_chunking"]["enabled"]:
            from prod.chunk_training_data import chunk_files

            if config["data_chunking"]["use_chunk_from_same_run"]:
                chunk_data_path = (
                    Path(config["data_pipeline"]["output_dir"])
                    / config["global_params"]["run_id"]
                )
            else:
                chunk_data_path = Path(config["data_chunking"]["input_data_path"])

            log.info("Chunking data")
            chunk_files(
                desired_sets=config["data_chunking"]["enabled_sets"],
                data_splits_names=config["data_chunking"]["enabled_splits"],
                input_data_dir=Path(config["data_chunking"]["input_data_path"]),
                output_data_dir=Path(config["data_chunking"]["output_data_path"]),
                file_chunk_sizes=config["data_chunking"]["chunk_size"],
                run_id=config["global_params"]["run_id"],
            )

        if config["training"]["enabled"]:
            from prod.train_model import train

            # verify_model_config(config) # TODO: fix this
            log.info("Training model")
            train(run=run)

        log.info("Run complete")
