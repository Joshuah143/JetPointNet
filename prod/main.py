from utils.dev_tools import load_config

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

    if config["training"]["enabled"]:
        from train_model import train

        print("Training model")
        train({})
