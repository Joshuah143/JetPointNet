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
import tensorflow.keras.metrics as metrics
import tensorflow.keras.backend as K
import wandb
from wandb.sdk import Config

from JetPointNet import (
    TF_SEED,
    PointNetSegmentation,
    masked_weighted_accuracy,
    masked_weighted_loss,
    set_global_determinism,
)
from numpy.lib import recfunctions as rfn
from tqdm.auto import tqdm
from wandb.sdk.wandb_run import Run
from loguru import logger as log

from utils.train_helpers import verify_model_config, setup_compute

# tf.config.run_functions_eagerly(True) - Useful when using the debugger - don't delete, but should not be used in production


# EXPERIMENT_NAME = config["training"]["experiment"]["experiment_name"]
# RESULTS_PATH = Path("result") / EXPERIMENT_NAME
# RESULTS_PATH.mkdir(exist_ok=True, parents=True)
# MODELS_PATH = Path("models") / EXPERIMENT_NAME
# MODELS_PATH.mkdir(exist_ok=True, parents=True)
#
# TRAINING_FILE_LOCATION = Path(config["training"]["infra"]["input_data_path"])
# MAX_SAMPLE_LENGTH = config["global_params"]["max_sample_length"]
# TRAIN_INPUT_SETS = config["training"]["hyperparameters"]["training_input_sets"]
# TRAIN_INPUTS = config["training"]["hyperparameters"]["training_labels"]
# TRAIN_TARGETS = config["training"]["hyperparameters"]["training_targets"]
#
# baseline_configuration = config["training"]["hyperparameters"]["model_params"]
# TRAIN_STEPS = VAL_STEPS = (
#     baseline_configuration["EPOCH_COMPLEXITY"] // baseline_configuration["BATCH_SIZE"],
# )


def load_data_from_npz(
    npz_file: Path, max_sample_length: int, train_inputs: list, train_targets: list
):
    all_feats = np.load(npz_file)  # may need ['feats] depending on numpy version
    feats = all_feats[:, :max_sample_length][
        train_inputs
    ]  # discard tracking information
    frac_labels = all_feats[:, :max_sample_length][train_targets]
    energy_weights = all_feats[:, :max_sample_length]["cell_E"]
    return feats, frac_labels, energy_weights


def _init_buffers():
    return [], [], []


def _format_batch(feats_buffer, targets_buffer, e_weights_buffer):
    batch_feats = np.array(feats_buffer)
    batch_targets = np.expand_dims(targets_buffer, axis=-1)
    batch_e_weights = np.expand_dims(e_weights_buffer, axis=-1)

    return batch_feats, batch_targets, batch_e_weights


def single_set_data_generator(
    data_dir: Path,
    set_name: str,
    batch_size: int,
    max_sample_length: int,
    train_inputs: list,
    train_targets: list,
    **kwargs,
):
    if kwargs.get("seed", 0):
        np.random.seed(kwargs["seed"])

    # get filenames and initialize buffers
    npz_files = glob.glob(
        os.path.join(data_dir, set_name, "*.npy")
    )  # TODO: switch back to NPZ after rename of files
    feats_buffer, targets_buffer, e_weights_buffer = _init_buffers()
    if len(npz_files) == 0:
        raise Exception(f"No npz files found for {set_name} in {data_dir}")

    while True:
        np.random.shuffle(npz_files)
        for npz_file in npz_files:

            # Read data chunk and initialize counters
            feats, targets, e_weights = load_data_from_npz(
                npz_file,
                max_sample_length=max_sample_length,
                train_inputs=train_inputs,
                train_targets=train_targets,
            )
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


def consistent_data_generator(
    data_dir: Path,
    data_sets: dict,
    batch_size: int,
    max_sample_length: int,
    train_inputs: list,
    train_targets: list,
):
    # Set up the generators
    generator_dict = {
        set_name: single_set_data_generator(
            data_dir,
            set_name,
            int(batch_size * inclusion_ratio),
            max_sample_length=max_sample_length,
            train_targets=train_targets,
            train_inputs=train_inputs,
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

    log.info("Total params: {:,}".format(trainable_count + non_trainable_count))
    log.info("Trainable params: {:,}".format(trainable_count))
    log.info("Non-trainable params: {:,}".format(non_trainable_count))
    return model, trainable_count


@tf.function
def train_step(
    x,
    y,
    energy_weights,
    model,
    loss_function,
    x_class,
    transform,
    weighted_accuracy_metric,
    unweighted_accuracy_metric,
):
    print(x.shape, y.shape, energy_weights.shape, x_class.shape)

    assert len(x.shape) == 3  # (batch_size, num_points, features)
    assert len(y.shape) == 3  # (batch_size, num_points, features)
    assert len(energy_weights.shape) == 2  # (batch_size, num_points)
    assert len(x_class.shape) == 2  # (batch_size, num_points)

    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = masked_weighted_loss(
            y_true=y,
            y_pred=predictions,
            energies=energy_weights,
            loss_function=loss_function,
            x_class=x_class,
            transform=transform,  # model_params.LOSS_ENERGY_WEIGHTING,
        )
        reg_acc, weighted_acc = masked_weighted_accuracy(
            y_true=y,
            y_pred=predictions,
            energies=energy_weights,
            x_class=x_class,
            weighted_accuracy_metric=weighted_accuracy_metric,
            unweighted_accuracy_metric=unweighted_accuracy_metric,
            transform=transform,
        )
    grads = tape.gradient(loss, model.trainable_variables)
    return loss, reg_acc, weighted_acc, grads


@tf.function
def val_step(
    x,
    y,
    energy_weights,
    model,
    loss_function,
    x_class,
    transform,
    weighted_accuracy_metric,
    unweighted_accuracy_metric,
):
    assert len(x.shape) == 3  # (batch_size, num_points, features)
    assert len(y.shape) == 3  # (batch_size, num_points, features)
    assert len(energy_weights.shape) == 2  # (batch_size, num_points)
    assert len(x_class.shape) == 2  # (batch_size, num_points)

    predictions = model(x, training=False)
    v_loss = masked_weighted_loss(
        y_true=y,
        y_pred=predictions,
        energies=energy_weights,
        transform=transform,
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
        transform=transform,
    )
    return v_loss, reg_acc, weighted_acc, predictions


def train(*, run: Run):
    run_config: Config = run.config
    setup_compute(dict(run_config))

    models_save_path = Path(run_config["training"]["infra"]["model_save_path"])

    train_steps = val_steps = (
        run_config["training"]["epoch_complexity"]
        // run_config["training"]["batch_size"]
    )
    log.debug(f"{train_steps = };\t{val_steps = }")

    seed = run_config["training"]["infra"]["TF_SEED"]
    log.debug(f"Setting training determinism based on {seed=}")
    set_global_determinism(seed=seed)

    train_inputs = run_config["training"]["params"]["training_labels"]
    train_targets = run_config["training"]["params"]["train_targets"]

    log.info(f"Training inputs: {train_inputs}")
    log.info(f"Training targets: {train_targets}")

    # model, trackers and callbacks and setup
    model, trainable_params = _setup_model(
        num_points=run_config["global_params"]["max_sample_length"],
        num_features=len(train_inputs),
        num_classes=len(train_targets),
        output_activation=run_config["training"]["hyperparameters"]["model_params"][
            "output_activation_function"
        ],
        model_version=run_config["training"]["hyperparameters"]["model_params"][
            "model_version"
        ],
    )

    wandb.log({"trainable_params": trainable_params})

    train_loss_tracker = metrics.Mean(name="train_loss")
    train_reg_acc = metrics.Mean(name="train_regular_accuracy")
    train_weighted_acc = metrics.Mean(name="train_weighted_accuracy")

    val_loss_tracker = metrics.Mean(name="val_loss")
    val_reg_acc = metrics.Mean(name="val_regular_accuracy")
    val_weighted_acc = metrics.Mean(name="val_weighted_accuracy")

    mean_iou_metric = tf.keras.metrics.OneHotMeanIoU(len(train_targets))
    val_weighted_f1_score = tf.keras.metrics.F1Score(
        threshold=run_config["training"]["hyperparameters"]["model_params"][
            "output_layer_segmentation_cutoff"
        ]
    )
    val_unweighted_f1_score = tf.keras.metrics.F1Score(
        threshold=run_config["training"]["hyperparameters"]["model_params"][
            "output_layer_segmentation_cutoff"
        ]
    )

    # Callbacks
    best_checkpoint_path = f"{models_save_path}/PointNet_best_name={run.name}.keras"

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_checkpoint_path,
        save_best_only=True,
        monitor=run_config["training"]["hyperparameters"]["model_params"][
            "primary_metric"
        ],  # Monitor validation loss
        mode=run_config["training"]["hyperparameters"]["model_params"][
            "primary_metric_mode"
        ],
        save_weights_only=False,
        verbose=1,
    )
    checkpoint_callback.set_model(model)

    # EarlyStopping
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor=run_config["training"]["hyperparameters"]["model_params"][
            "primary_metric"
        ],  # "val_weighted_accuracy",  # Monitor validation loss
        mode=run_config["training"]["hyperparameters"]["model_params"][
            "primary_metric_mode"
        ],  # "max",  # Trigger when validation loss stops decreasing
        patience=run_config["training"]["hyperparameters"]["model_params"][
            "early_stopping_patience"
        ],  # Number of epochs to wait before stopping if no improvement
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
            run_config["training"]["hyperparameters"]["model_params"]["learning_rate"]
        ),
        decay_steps=train_steps,
        decay_rate=(
            run_config["training"]["hyperparameters"]["model_params"][
                "learning_rate_decay"
            ]
        ),
    )

    # Optimizer & Loss
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=run_config["training"]["hyperparameters"]["model_params"][
            "learning_rate_beta_1"
        ],
        beta_2=run_config["training"]["hyperparameters"]["model_params"][
            "learning_rate_beta_2"
        ],
        # decay=config.LR_DECAY,
    )

    # Will raise AttributeError if the loss function is not found
    logits = (
        run_config["training"]["hyperparameters"]["model_params"][
            "output_activation_function"
        ]
        == "linear"
    )
    loss_function = getattr(
        tf.keras.losses,
        run_config["training"]["hyperparameters"]["model_params"]["loss_function"],
    )(
        from_logits=logits,  # NOTE: False for "sigmoid", True for "linear"
        # reduction='none', # TODO: check if this is needed, look into the loss funct params again
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

    for epoch in range(run_config["training"]["epochs"]):
        log.info(f"\nStart of epoch {epoch}")
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
            Path(run_config["training"]["infra"]["input_data_path"]) / "train",
            run_config["training"]["hyperparameters"]["training_input_sets"],
            run_config["training"]["batch_size"],
            max_sample_length=run_config["global_params"]["max_sample_length"],
            train_inputs=train_inputs,
            train_targets=train_targets,
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
            e_weight_train = np.squeeze(e_weight_train, axis=-1)
            y_batch_train = np.squeeze(y_batch_train, axis=-1)

            if step >= train_steps:
                break
            loss_value, reg_acc_value, weighted_acc_value, grads = train_step(
                x_batch_train,
                y_batch_train,
                e_weight_train,
                model,
                loss_function,
                x_catagories,
                transform=run_config["training"]["hyperparameters"]["model_params"][
                    "energy_weighting_transform"
                ],
                weighted_accuracy_metric=weighted_accuracy_metric,
                unweighted_accuracy_metric=unweighted_accuracy_metric,
            )
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss_tracker.update_state(loss_value)
            train_reg_acc.update_state(reg_acc_value)
            train_weighted_acc.update_state(weighted_acc_value)

            log.info(
                f"\rEpoch {epoch + 1}, Step {step + 1}/{train_steps}, "
                f"Training Loss: {train_loss_tracker.result().numpy():.4e}, "
                f"Reg Acc: {train_reg_acc.result().numpy():.4f}, "
                f"Weighted Acc: {train_weighted_acc.result().numpy():.4f}",
                end="",
            )
            batch_loss_train.append(train_loss_tracker.result().numpy())
            batch_accuracy_train.append(train_reg_acc.result().numpy())
            batch_weighted_accuracy_train.append(train_weighted_acc.result())

        log.info(f"\nTraining loss over epoch: {train_loss_tracker.result():.4e}")
        log.info(f"\nTime taken for training: {time.time() - start_time:.2f} sec")

        batch_loss_val, batch_accuracy_val, batch_weighted_accuracy_val = [], [], []

        _val_generator = consistent_data_generator(
            Path(run_config["training"]["infra"]["input_data_path"]) / "val",
            run_config["training"]["hyperparameters"]["training_input_sets"],
            run_config["training"]["batch_size"],
            max_sample_length=run_config["global_params"]["max_sample_length"],
            train_inputs=train_inputs,
            train_targets=train_targets,
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
            y_batch_val = np.squeeze(y_batch_val, axis=-1)
            e_weight_val = np.squeeze(e_weight_val, axis=-1)

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
                transform=run_config["training"]["hyperparameters"]["model_params"][
                    "energy_weighting_transform"
                ],
                weighted_accuracy_metric=weighted_accuracy_metric,
                unweighted_accuracy_metric=unweighted_accuracy_metric,
            )
            val_loss_tracker.update_state(val_loss_value)
            val_reg_acc.update_state(val_reg_acc_value)
            val_weighted_acc.update_state(val_weighted_acc_value)

            mask = x_batch_val_named["category"] == 1  # remove non-energy points
            val_true_labels.extend(y_batch_val[mask])
            val_energy_weights.extend(e_weight_val[mask])
            val_predictions.extend(predicted_y.numpy()[mask])

            log.info(
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

        log.info(f"\nValidation F1 Score: {val_f1}")
        log.info(f"\nValidation loss: {val_loss_tracker.result():.4e}")
        log.info(f"\nTime taken for validation: {time.time() - start_time:.2f} sec")

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

        for i in range(len(train_targets)):
            performance[f"val/f1_score_{train_targets[i]}"] = val_f1[i]
            performance[f"val/f1_weighted_score_{train_targets[i]}"] = weighted_val_f1[
                i
            ]

        wandb.log(performance)

        # callbacks

        # discard first epochs to trigger callbacks
        if epoch > 50:
            if run_config["training"]["hyperparameters"]["model_params"][
                "save_intermediates"
            ] and (
                epoch
                % run_config["training"]["hyperparameters"]["model_params"]["save_freq"]
                == 0
            ):
                checkpoint_path = (
                    f"{models_save_path}/PointNet_{epoch=}_name={run.name}.keras"
                )
                model.save(checkpoint_path)
                checkpoint_callback.on_epoch_end(epoch, logs=performance)

            if run_config["training"]["hyperparameters"]["model_params"][
                "early_stopping"
            ]:
                early_stopping_callback.on_epoch_end(epoch, logs=performance)
                if early_stopping_callback.model.stop_training:
                    log.info(f"Early stopping triggered at epoch {epoch}")
                    break

    last_checkpoint_path = (
        f"{models_save_path}/PointNet_last_{epoch=}_name={run.name}.keras"
    )
    model.save(last_checkpoint_path)

    # Log the best and last models to wandb
    best_model_artifact = wandb.Artifact("best_baseline", type="model")
    best_model_artifact.add_file(best_checkpoint_path)
    wandb.log_artifact(best_model_artifact)

    final_model_artifact = wandb.Artifact("last_epoch_baseline", type="model")
    final_model_artifact.add_file(last_checkpoint_path)
    wandb.log_artifact(final_model_artifact)
