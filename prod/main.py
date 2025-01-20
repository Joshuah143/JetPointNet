from utils.dev_tools import load_config
from pathlib import Path

config = load_config()

if __name__ == "__main__":
    if config["data_pipeline"]["enabled"]:
        match config["data_pipeline"]["pipeline"]:
            case "overlapping":
                from produce_overlapping_training_data import save_train_data

                print("Processing overlapping data pipeline")
                save_train_data()
            case "augmented":
                from produce_augmented_training_data import save_train_data

                print("Processing augmented data pipeline")
                save_train_data()
            case _:
                print("Invalid pipeline type")
                raise ValueError("Invalid pipeline type")

    if config["data_chunking"]["enabled"]:
        from chunk_training_data import chunk_files

        print("Chunking data")
        chunk_files(
            desired_sets=config["data_chunking"]["enabled_sets"],
            data_splits_names=config["data_chunking"]["enabled_splits"],
            input_data_dir=Path(config["data_chunking"]["input_data_path"]),
            output_data_dir=Path(config["data_chunking"]["output_data_path"]),
            file_chunk_sizes=config["data_chunking"]["chunk_size"],
        )

    if config["training"]["enabled"]:
        from train_model import train

        print("Training model")
        train()
