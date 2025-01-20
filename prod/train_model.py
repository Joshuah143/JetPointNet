# TODO:
# - implement logging weights and gradients
# - implement logging confusion matrix: MINOR
# - implement lr scheduler: DONE, but need experimenting with more schedulers
# - move to .fit instead of custom train/val loop
# - experiment with more losses/metrics: ATTEMPTED,
# doesn't seem feasible because of per-point weighted loss (can't pass weights to loss during .fit) --> TO CHECK BETTER?

import glob
import math
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.metrics as metrics
import tensorflow.keras.backend as K
import wandb
from JetPointNet import (
    TF_SEED,
    PointNetSegmentation,
    masked_weighted_accuracy,
    masked_weighted_loss,
    set_global_determinism,
)
from numpy.lib import recfunctions as rfn
from tqdm.auto import tqdm

from prod.utils.train_helpers import verify_model_config, setup_compute
from utils.dev_tools import load_config

# tf.config.run_functions_eagerly(True) - Useful when using the debugger - don't delete, but should not be used in production

config = load_config()
setup_compute(config)

if not config["training"]["enabled"]:
    raise Exception("Training is disabled in the config.")

EXPERIMENT_NAME = config["training"]["experiment"]["experiment_name"]
RESULTS_PATH = Path("result") / EXPERIMENT_NAME
RESULTS_PATH.mkdir(exist_ok=True, parents=True)
MODELS_PATH = Path("models") / EXPERIMENT_NAME
MODELS_PATH.mkdir(exist_ok=True, parents=True)

TRAINING_FILE_LOCATION = Path(config["training"]["infra"]["input_data_path"])
MAX_SAMPLE_LENGTH = config["global_params"]["max_sample_length"]
TRAIN_INPUT_SETS = config["training"]["hyperparameters"]["training_input_sets"]
TRAIN_INPUTS = config["training"]["hyperparameters"]["training_labels"]
TRAIN_TARGETS = config["training"]["hyperparameters"]["training_targets"]

baseline_configuration = config["training"]["hyperparameters"]["model_params"]
TRAIN_STEPS = VAL_STEPS = (
    baseline_configuration["EPOCH_COMPLEXITY"] // baseline_configuration["BATCH_SIZE"],
)

verify_model_config(config)


def load_data_from_npz(npz_file):
    all_feats = np.load(npz_file)  # may need ['feats] depending on numpy version
    feats = all_feats[:, :MAX_SAMPLE_LENGTH][
        TRAIN_INPUTS
    ]  # discard tracking information
    frac_labels = all_feats[:, :MAX_SAMPLE_LENGTH][TRAIN_TARGETS]
    energy_weights = all_feats[:, :MAX_SAMPLE_LENGTH]["cell_E"]
    return feats, frac_labels, energy_weights


def _init_buffers():
    return [], [], []


def _format_batch(feats_buffer, targets_buffer, e_weights_buffer):
    batch_feats = np.array(feats_buffer)
    batch_targets = np.expand_dims(targets_buffer, axis=-1)
    batch_e_weights = np.expand_dims(e_weights_buffer, axis=-1)

    return batch_feats, batch_targets, batch_e_weights


def single_set_data_generator(data_dir, set_name, batch_size: int, **kwargs):
    if kwargs.get("seed", 0):
        np.random.seed(kwargs["seed"])

    # get filenames and initialize buffers
    npz_files = glob.glob(
        os.path.join(data_dir, set_name, "*.np[yz]")
    )  # TODO: switch back to NPZ after rename of files
    feats_buffer, targets_buffer, e_weights_buffer = _init_buffers()
    if len(npz_files) == 0:
        raise Exception(f"No npz files found for {set_name} in {data_dir}")

    while True:
        np.random.shuffle(npz_files)
        for npz_file in npz_files:

            # Read data chunk and initialize counters
            feats, targets, e_weights = load_data_from_npz(npz_file)
            file_size = feats.shape[0]
            # initially all data are still not used: can fill full file in batch starting at index 0
            unprocessed_size = file_size
            last_batch_idx = 0
            fill_size = batch_size - len(feats_buffer)

            # loop through chunk until all points are processed
            while last_batch_idx < file_size:

                # get N of elements remaining to reach to batch_size
                fill_size = batch_size - len(feats_buffer)
                # get fill_size elements starting from last_batch_idx
                last_index = min(last_batch_idx + fill_size, file_size)
                feats_buffer.extend(feats[last_batch_idx:last_index])
                targets_buffer.extend(targets[last_batch_idx:last_index])
                e_weights_buffer.extend(e_weights[last_batch_idx:last_index])

                # update unprocessed points and last index
                unprocessed_size -= fill_size
                last_batch_idx += fill_size

                # check if batch is full, in case yield + reset buffers
                if len(feats_buffer) == batch_size:
                    batch_feats, batch_targets, batch_e_weights = (
                        feats_buffer,
                        targets_buffer,
                        e_weights_buffer,
                    )
                    feats_buffer, targets_buffer, e_weights_buffer = _init_buffers()
                    yield batch_feats, batch_targets, batch_e_weights


def consistent_data_generator(data_dir, data_sets: dict, batch_size: int, **kwargs):
    # Set up the generators
    generator_dict = {
        set_name: single_set_data_generator(
            data_dir, set_name, int(batch_size * inclusion_ratio)
        )
        for set_name, inclusion_ratio in data_sets.items()
    }
    while True:
        feats_buffer, targets_buffer, e_weights_buffer = _init_buffers()
        for generator in generator_dict.values():
            feats_inner_buffer, targets_inner_buffer, e_weights_inner_buffer = next(
                generator
            )
            feats_buffer.extend(feats_inner_buffer)
            targets_buffer.extend(targets_inner_buffer)
            e_weights_buffer.extend(e_weights_inner_buffer)

        yield _format_batch(feats_buffer, targets_buffer, e_weights_buffer)


def calculate_steps(data_dir, batch_size):
    total_samples = 0
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    for npz_file in tqdm(npz_files):
        data = np.load(npz_file)
        total_samples += data["feats"].shape[0]
    return math.ceil(total_samples / batch_size)


def _setup_model(
    num_points: int,
    num_features: int,
    output_activation: str,
    model_version: int,
    num_classes: int = 1,
):
    model = PointNetSegmentation(
        num_points=num_points,
        num_features=num_features,
        num_classes=num_classes,
        output_activation_function=output_activation,
        model_version=model_version,
    )
    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum(
        [K.count_params(w) for w in model.non_trainable_weights]
    )

    print("Total params: {:,}".format(trainable_count + non_trainable_count))
    print("Trainable params: {:,}".format(trainable_count))
    print("Non-trainable params: {:,}".format(non_trainable_count))
    return model, trainable_count


def train():
    run_config = baseline_configuration
    with wandb.init(
        project="pointcloud",
        config=run_config,
        job_type="training",
        # tags=[TRAIN_OUTPUT_DIRECTORY_NAME,
        #       TRAIN_DATASET_NAME,
        #       str(TRAIN_ALlOWED_SETS.keys())],
        notes="",
    ) as run:
        model_params = wandb.config

        # number of steps and seed
        train_steps = (
            model_params.TRAIN_STEPS
        )  # calculate_steps(TRAIN_DIR, config.BATCH_SIZE)  # 47
        val_steps = (
            model_params.VAL_STEPS
        )  # calculate_steps(VAL_DIR, config.BATCH_SIZE)  # 26
        print(f"{train_steps = };\t{val_steps = }")

        seed = model_params.TF_SEED
        print(f"Setting training determinism based on {seed=}")
        set_global_determinism(seed=seed)

        # training and validation steps
        @tf.function
        def train_step(x, y, energy_weights, model, loss_function, x_class):
            with tf.GradientTape() as tape:
                predictions = model(x, training=True)
                loss = masked_weighted_loss(
                    y_true=y,
                    y_pred=predictions,
                    energies=energy_weights,
                    loss_function=loss_function,
                    x_class=x_class,
                    transform=model_params.LOSS_ENERGY_WEIGHTING,
                )
                reg_acc, weighted_acc = masked_weighted_accuracy(
                    y_true=y,
                    y_pred=predictions,
                    energies=energy_weights,
                    x_class=x_class,
                    weighted_accuracy_metric=weighted_accuracy_metric,
                    unweighted_accuracy_metric=unweighted_accuracy_metric,
                    transform=model_params.ACC_ENERGY_WEIGHTING,
                )
            grads = tape.gradient(loss, model.trainable_variables)
            return loss, reg_acc, weighted_acc, grads

        @tf.function
        def val_step(x, y, energy_weights, model, loss_function, x_class):
            predictions = model(x, training=False)
            v_loss = masked_weighted_loss(
                y_true=y,
                y_pred=predictions,
                energies=energy_weights,
                transform=model_params.LOSS_ENERGY_WEIGHTING,
                x_class=x_class,
                loss_function=loss_function,
            )
            reg_acc, weighted_acc = masked_weighted_accuracy(
                y_true=y,
                y_pred=predictions,
                energies=energy_weights,
                x_class=x_class,
                weighted_accuracy_metric=weighted_accuracy_metric,
                unweighted_accuracy_metric=unweighted_accuracy_metric,
                transform=model_params.ACC_ENERGY_WEIGHTING,
            )
            return v_loss, reg_acc, weighted_acc, predictions

        # model, trackers and callbacks and setup
        model, trainable_params = _setup_model(
            num_points=model_params.MAX_SAMPLE_LENGTH,
            num_features=len(TRAIN_INPUTS),
            num_classes=len(TRAIN_TARGETS),
            output_activation=model_params.OUTPUT_ACTIVATION_FUNCTION,
            model_version=model_params.MODEL_VERSION,
        )

        wandb.log({"trainable_params": trainable_params})

        train_loss_tracker = metrics.Mean(name="train_loss")
        train_reg_acc = metrics.Mean(name="train_regular_accuracy")
        train_weighted_acc = metrics.Mean(name="train_weighted_accuracy")

        val_loss_tracker = metrics.Mean(name="val_loss")
        val_reg_acc = metrics.Mean(name="val_regular_accuracy")
        val_weighted_acc = metrics.Mean(name="val_weighted_accuracy")
        mean_iou_metric = tf.keras.metrics.OneHotMeanIoU(len(TRAIN_TARGETS))
        val_weighted_f1_score = tf.keras.metrics.F1Score(
            threshold=model_params.OUTPUT_LAYER_SEGMENTATION_CUTOFF
        )
        val_unweighted_f1_score = tf.keras.metrics.F1Score(
            threshold=model_params.OUTPUT_LAYER_SEGMENTATION_CUTOFF
        )

        # Callbacks
        # ModelCheckpoint
        best_checkpoint_path = f"{MODELS_PATH}/PointNet_best_name={run.name}.keras"

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=best_checkpoint_path,
            save_best_only=True,
            monitor=model_params.METRIC,  # Monitor validation loss
            mode=model_params.MODE,  # Save the model with the minimum validation loss
            save_weights_only=False,
            verbose=1,
        )
        checkpoint_callback.set_model(model)

        # EarlyStopping
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor=model_params.METRIC,  # "val_weighted_accuracy",  # Monitor validation loss
            mode=model_params.MODE,  # "max",  # Trigger when validation loss stops decreasing
            patience=model_params.ES_PATIENCE,  # Number of epochs to wait before stopping if no improvement
            verbose=1,
        )
        early_stopping_callback.set_model(model)

        # Learning Rate Scheduler
        # lr_callback = CustomLRScheduler(
        #     optim_lr=optimizer.learning_rate,
        #     lr_max=config.LR_MAX * train_steps * config.BATCH_SIZE,
        #     lr_min=config.LR_MIN,
        #     lr_ramp_ep=config.LR_RAMP_EP,
        #     lr_sus_ep=config.LR_SUS_EP,
        #     lr_decay=config.LR_DECAY,
        #     verbose=1,
        # )

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=(
                model_params.TRAIN_LR
                if not model_params.IS_TUNE
                else model_params.TUNE_LR
            ),
            decay_steps=train_steps,
            decay_rate=(
                model_params.TRAIN_LR_DECAY
                if not model_params.IS_TUNE
                else model_params.TUNE_LR_DECAY
            ),
        )

        # Optimizer & Loss
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=model_params.LR_BETA1,
            beta_2=model_params.LR_BETA2,
            # decay=config.LR_DECAY,
        )

        # Will raise AttributeError if the loss function is not found
        logits = model_params.OUTPUT_ACTIVATION_FUNCTION == "linear"
        loss_function = getattr(tf.keras.losses, model_params.LOSS_FUNCTION)(
            # from_logits=logits,  # NOTE: False for "sigmoid", True for "linear"
            # reduction='none',
        )

        weighted_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
        unweighted_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

        # NOTE: the match/case below may still be useful in case of differential processing depending on the chosen loss function
        # match config.LOSS_FUNCTION:
        #     case "BCE":
        #         loss_function = tf.keras.losses.BinaryCrossentropy(
        #             from_logits=False
        #         )
        #     case "FocalBCE":
        #         loss_function = tf.keras.losses.BinaryFocalCrossentropy(
        #             from_logits=False
        #         )
        #     case _:
        # raise Exception("Undefined Loss Function")

        for epoch in range(model_params.EPOCHS):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()

            # update LR with scheduler
            # lr_callback.on_epoch_begin(epoch)

            train_loss_tracker.reset_state()
            train_reg_acc.reset_state()
            train_weighted_acc.reset_state()

            val_loss_tracker.reset_state()
            val_reg_acc.reset_state()
            val_weighted_acc.reset_state()

            val_weighted_f1_score.reset_state()
            val_unweighted_f1_score.reset_state()
            mean_iou_metric.reset_state()

            val_true_labels = []
            val_predictions = []
            val_energy_weights = []

            # train step
            batch_loss_train, batch_accuracy_train, batch_weighted_accuracy_train = (
                [],
                [],
                [],
            )

            _train_generator = consistent_data_generator(
                TRAINING_FILE_LOCATION / "train",
                model_params.INPUT_SETS,
                model_params.BATCH_SIZE,
            )
            train_generator = enumerate(_train_generator)

            for step, (
                x_batch_train_named,
                y_batch_train,
                e_weight_train,
            ) in train_generator:
                x_catagories = x_batch_train_named["category"]
                x_batch_train = rfn.structured_to_unstructured(x_batch_train_named)
                y_batch_train = rfn.structured_to_unstructured(y_batch_train)
                # For some reason the second last dim is always 1, not sure why but this fixes it
                y_batch_train = np.squeeze(y_batch_train)
                e_weight_train = np.squeeze(e_weight_train)

                if step >= train_steps:
                    break
                loss_value, reg_acc_value, weighted_acc_value, grads = train_step(
                    x_batch_train,
                    y_batch_train,
                    e_weight_train,
                    model,
                    loss_function,
                    x_catagories,
                )
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                train_loss_tracker.update_state(loss_value)
                train_reg_acc.update_state(reg_acc_value)
                train_weighted_acc.update_state(weighted_acc_value)

                print(
                    f"\rEpoch {epoch + 1}, Step {step + 1}/{train_steps}, "
                    f"Training Loss: {train_loss_tracker.result().numpy():.4e}, "
                    f"Reg Acc: {train_reg_acc.result().numpy():.4f}, "
                    f"Weighted Acc: {train_weighted_acc.result().numpy():.4f}",
                    end="",
                )
                batch_loss_train.append(train_loss_tracker.result().numpy())
                batch_accuracy_train.append(train_reg_acc.result().numpy())
                batch_weighted_accuracy_train.append(train_weighted_acc.result())

            print(f"\nTraining loss over epoch: {train_loss_tracker.result():.4e}")
            print(f"\nTime taken for training: {time.time() - start_time:.2f} sec")

            batch_loss_val, batch_accuracy_val, batch_weighted_accuracy_val = [], [], []

            _val_generator = consistent_data_generator(
                TRAINING_FILE_LOCATION / "val",
                model_params.INPUT_SETS,
                model_params.BATCH_SIZE,
            )
            val_generator = enumerate(_val_generator)

            for step, (
                x_batch_val_named,
                y_batch_val_named,
                e_weight_val,
            ) in val_generator:
                x_catagories_val = x_batch_val_named["category"]
                x_batch_val = rfn.structured_to_unstructured(x_batch_val_named)
                y_batch_val = rfn.structured_to_unstructured(y_batch_val_named)
                # For some reason the second last dim is always 1, not sure why but this fixes it
                y_batch_val = np.squeeze(y_batch_val)
                e_weight_val = np.squeeze(e_weight_val)
                if step >= val_steps:
                    break
                (
                    val_loss_value,
                    val_reg_acc_value,
                    val_weighted_acc_value,
                    predicted_y,
                ) = val_step(
                    x_batch_val,
                    y_batch_val,
                    e_weight_val,
                    model,
                    loss_function,
                    x_catagories_val,
                )
                val_loss_tracker.update_state(val_loss_value)
                val_reg_acc.update_state(val_reg_acc_value)
                val_weighted_acc.update_state(val_weighted_acc_value)

                mask = x_batch_val_named["category"] == 1  # remove non-energy points
                val_true_labels.extend(y_batch_val[mask])
                val_energy_weights.extend(e_weight_val[mask])
                val_predictions.extend(predicted_y.numpy()[mask])

                print(
                    f"\rEpoch {epoch + 1}, Step {step + 1}/{val_steps}, "
                    f"Validation Loss: {val_loss_tracker.result().numpy():.4e}, "
                    f"Reg Acc: {val_reg_acc.result().numpy():.4f}, "
                    f"Weighted Acc: {val_weighted_acc.result().numpy():.4f}",
                    end="",
                )

                batch_loss_val.append(val_loss_tracker.result().numpy())
                batch_accuracy_val.append(val_reg_acc.result().numpy())
                batch_weighted_accuracy_val.append(val_weighted_acc.result())

            val_true_labels = tf.convert_to_tensor(val_true_labels)
            val_predictions = tf.convert_to_tensor(val_predictions)
            val_weights = tf.convert_to_tensor(val_energy_weights)

            val_true_labels = tf.cast(val_true_labels, dtype=tf.float32)
            val_predictions = tf.cast(val_predictions, dtype=tf.float32)
            val_weights = tf.cast(val_weights, dtype=tf.float32)

            val_weighted_f1_score.update_state(
                tf.expand_dims(
                    val_true_labels, axis=-1
                ),  # tf.expand_dims(val_true_labels, axis=-1),
                val_predictions,  # tf.expand_dims(val_predictions, axis=-1),
                sample_weight=val_weights,
            )
            val_unweighted_f1_score.update_state(
                tf.expand_dims(val_true_labels, axis=-1),
                val_predictions,  # tf.expand_dims(val_predictions, axis=-1),
            )
            mean_iou_metric.update_state(
                tf.expand_dims(val_true_labels, axis=-1), val_predictions
            )

            val_f1 = val_unweighted_f1_score.result().numpy()
            weighted_val_f1 = val_weighted_f1_score.result().numpy()

            print(f"\nValidation F1 Score: {val_f1}")
            print(f"\nValidation loss: {val_loss_tracker.result():.4e}")
            print(f"\nTime taken for validation: {time.time() - start_time:.2f} sec")

            performance = {
                "epoch": epoch,
                "train/loss": train_loss_tracker.result().numpy(),
                "train/accuracy": train_reg_acc.result().numpy(),
                "train/weighted_accuracy": train_weighted_acc.result().numpy(),
                "val/loss": val_loss_tracker.result().numpy(),
                "val/accuracy": val_reg_acc.result().numpy(),
                "val/weighted_accuracy": val_weighted_acc.result().numpy(),
                "learning_rate": optimizer.learning_rate.numpy(),
                "val/mean_iou": mean_iou_metric.result().numpy(),
            }

            for i in range(len(TRAIN_TARGETS)):
                performance[f"val/f1_score_{TRAIN_TARGETS[i]}"] = val_f1[i]
                performance[f"val/f1_weighted_score_{TRAIN_TARGETS[i]}"] = (
                    weighted_val_f1[i]
                )

            wandb.log(performance)

            # callbacks

            # discard first epochs to trigger callbacks
            if epoch > 50:
                if (
                    model_params.SAVE_INTERMEDIATES
                    and epoch % model_params.SAVE_FREQ == 0
                ):
                    checkpoint_path = (
                        f"{MODELS_PATH}/PointNet_{epoch=}_name={run.name}.keras"
                    )
                    model.save(checkpoint_path)
                    checkpoint_callback.on_epoch_end(epoch, logs=performance)

                if model_params.EARLY_STOPPING:
                    early_stopping_callback.on_epoch_end(epoch, logs=performance)
                    if early_stopping_callback.model.stop_training:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break

        print("\n\nTraining completed!")

        last_checkpoint_path = (
            f"{MODELS_PATH}/PointNet_last_{epoch=}_name={run.name}.keras"
        )
        model.save(last_checkpoint_path)

        # Log the best and last models to wandb
        best_model_artifact = wandb.Artifact("best_baseline", type="model")
        best_model_artifact.add_file(best_checkpoint_path)
        wandb.log_artifact(best_model_artifact)

        final_model_artifact = wandb.Artifact("last_epoch_baseline", type="model")
        final_model_artifact.add_file(last_checkpoint_path)
        wandb.log_artifact(final_model_artifact)


if __name__ == "__main__":
    train()


## OLD METHODS

# def progressive_data_generator(data_dir, simple_sets: list, complex_sets: list, batch_size: int, epoch: int,
#                                linear_decay: int, memory: int, **kwargs):
#     # Calculate the percentage of simple data to use
#     percent_simple_data = max(1 - epoch * linear_decay, memory)
#
#     # Setup the data generators for simple and complex datasets
#     simple_genorator = consistant_data_generator(data_dir,
#                                                  {set_name: percent_simple_data / len(simple_sets) for set_name in
#                                                   simple_sets}, int(batch_size * percent_simple_data), **kwargs)
#     complex_genorator = consistant_data_generator(data_dir,
#                                                   {set_name: (1 - percent_simple_data) / len(complex_sets) for
#                                                    set_name in complex_sets},
#                                                   int(batch_size * (1 - percent_simple_data)), **kwargs)
#     while True:
#         # Fetch the next batch from both generators
#         simple_feats, simple_targets, simple_e_weights = next(simple_genorator)
#         complex_feats, complex_targets, complex_e_weights = next(complex_genorator)
#
#         if simple_feats.size == 0:
#             total_feats, total_targets, total_e_weights = complex_feats, complex_targets, complex_e_weights
#         elif complex_feats.size == 0:
#             total_feats, total_targets, total_e_weights = simple_feats, simple_targets, simple_e_weights
#         else:
#             # Combine the features, targets, and weights from both generators
#             total_feats = np.concatenate([simple_feats, complex_feats], axis=0)
#             total_targets = np.concatenate([simple_targets, complex_targets], axis=0)
#             total_e_weights = np.concatenate([simple_e_weights, complex_e_weights], axis=0)
#
#         yield total_feats, total_targets, total_e_weights
