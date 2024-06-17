# TODO:
# - fix data generator: MINOR PRIORITY (last bit discarded, try to include into next batch)
# - implement logging weights and gradients
# - implement logging confusion matrix: MINOR
# - implement lr scheduler: DONE, but need experimenting with more schedulers
# - move to .fit instead of custom train/val loop
# - experiment with more losses/metrics: ATTEMPTED, doesn't seem feasible because of per-point weighted loss (can't pass weights to loss during .fit) --> TO CHECK BETTER?

import sys
from pathlib import Path
import os
import json

REPO_PATH = Path.home() / "workspace/jetpointnet"
SCRIPT_PATH = REPO_PATH / "python_scripts/data_processing/jets"
sys.path.append(str(SCRIPT_PATH))
SCRIPT_PATH = REPO_PATH / "python_scripts"
sys.path.append(str(SCRIPT_PATH))

import numpy as np
import tensorflow as tf
import glob
import math
import wandb
import time
import wandb
from tqdm.auto import tqdm
from numpy.lib import recfunctions as rfn
import tensorflow.keras.backend as K
from jets_training.models.JetPointNet import (
    PointNetSegmentation,
    masked_weighted_loss,
    masked_regular_accuracy,
    masked_weighted_accuracy,
    set_global_determinism,
    TF_SEED,
    CustomLRScheduler,
)
from data_processing.jets.preprocessing_header import (
    MAX_DISTANCE,
    NPZ_SAVE_LOC,
    ENERGY_SCALE,
)

# tf.config.run_functions_eagerly(True) - Useful when using the debugger - dont delete, but should not be used in production

# SET PATHS FOR I/O AND CONFIG
USER = Path.home().name
print(f"Logged in as {USER}")
if USER == "jhimmens":
    OUTPUT_DIRECTORY_NAME = "2000_events_w_fixed_hits"
    DATASET_NAME = "large_R"
    GPU_ID = "1"
elif USER == "luclissa":
    OUTPUT_DIRECTORY_NAME = "ttbar"
    DATASET_NAME = "benchmark"
    GPU_ID = "0"
else:
    raise Exception("UNKOWN USER")


EXPERIMENT_NAME = f"{OUTPUT_DIRECTORY_NAME}/{DATASET_NAME}"
RESULTS_PATH = REPO_PATH / "result" / EXPERIMENT_NAME
RESULTS_PATH.mkdir(exist_ok=True, parents=True)
MODELS_PATH = REPO_PATH / "models" / EXPERIMENT_NAME
MODELS_PATH.mkdir(exist_ok=True, parents=True)

SWEEP = True
RUN_SWEEP = True
SPLIT_SEED = 62
MAX_SAMPLE_LENGTH = 859  # 278 for delta R of 0.1, 859 for 0.2
BATCH_SIZE = 480
EPOCHS = 120
LR = 0.001
ES_PATIENCE = 15
TRAIN_DIR = NPZ_SAVE_LOC / "train"
VAL_DIR = NPZ_SAVE_LOC / "val"
ACC_ENERGY_WEIGHTING = "square"
LOSS_ENERGY_WEIGHTING = "square"
OUTPUT_ACTIVATION_FUNCTION = "sigmoid"  # softmax, linear (requires changes to the BCE fucntion in the loss function)
FRACTIONAL_ENERGY_CUTOFF = 0.5
OUTPUT_LAYER_SEGMENTATION_CUTOFF = 0.5

# POTENTIALLY OVERWRITTEN BY THE WANDB SWEEP:
LR_MAX = 0.000015 * 100 * BATCH_SIZE
LR_MIN = 1e-5
LR_RAMP_EP = 2
LR_SUS_EP = 10
LR_DECAY = 0.7

# note that if you change the output activation function you must change the loss function
if (
    OUTPUT_ACTIVATION_FUNCTION in ["softmax", "sigmoid"]
    and OUTPUT_LAYER_SEGMENTATION_CUTOFF != 0.5
):
    raise Exception("Invalid OUTPUT_LAYER_SEGMENTATION_CUTOFF")
if OUTPUT_ACTIVATION_FUNCTION in ["linear"] and OUTPUT_LAYER_SEGMENTATION_CUTOFF != 0:
    raise Exception("Invalid OUTPUT_LAYER_SEGMENTATION_CUTOFF")

TRAIN_INPUTS = [
    #'event_number',
    #'cell_ID',
    #'track_ID',
    #'delta_R',
    "category",
    "track_num",
    "normalized_x",
    "normalized_y",
    "normalized_z",
    "normalized_distance",
    "cell_E",
    "track_pt",
]


def load_data_from_npz(npz_file):
    data = np.load(npz_file)
    feats = data["feats"][:, :MAX_SAMPLE_LENGTH][
        TRAIN_INPUTS
    ]  # discard tracking information
    frac_labels = data["frac_labels"][:, :MAX_SAMPLE_LENGTH]
    energy_weights = data["tot_truth_e"][:, :MAX_SAMPLE_LENGTH]
    return feats, frac_labels, energy_weights


# NOTE: works with BATCH_SIZE but don't fix last batch size issue
def data_generator(data_dir, batch_size, drop_last=True):
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    if len(npz_files) == 0:
        raise Exception(f"{data_dir} does not contain npz files!")

    np.random.shuffle(npz_files)
    for npz_file in npz_files:
        feats, frac_labels, e_weights = load_data_from_npz(npz_file)
        dataset_size = feats.shape[0]
        for i in range(0, dataset_size, batch_size):
            end_index = i + batch_size
            if end_index > dataset_size:
                if drop_last:
                    continue  # Drop last smaller batch
                else:
                    batch_feats = feats[i:]
                    batch_labels = frac_labels[i:]
                    batch_e_weights = e_weights[i:]
            else:
                batch_feats = feats[i:end_index]
                batch_labels = frac_labels[i:end_index]
                batch_e_weights = e_weights[i:end_index]

            yield (
                batch_feats,
                batch_labels.reshape(*batch_labels.shape, 1),
                batch_e_weights.reshape(*batch_e_weights.shape, 1),
            )


def calculate_steps(data_dir, batch_size):
    total_samples = 0
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    for npz_file in tqdm(npz_files):
        data = np.load(npz_file)
        total_samples += data["feats"].shape[0]
    return math.ceil(total_samples / batch_size)


@tf.function
def train_step(x, y, energy_weights, model, optimizer, loss_function):
    print("traing step got here")
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = masked_weighted_loss(
            y_true=y,
            y_pred=predictions,
            energies=energy_weights,
            loss_function=loss_function,
            transform=wandb.config.loss_energy_weight_scheme,
            fractional_energy_cutoff=FRACTIONAL_ENERGY_CUTOFF,
        )
        reg_acc = masked_regular_accuracy(
            y_true=y,
            y_pred=predictions,
            output_layer_segmentation_cutoff=OUTPUT_LAYER_SEGMENTATION_CUTOFF,
            fractional_energy_cutoff=FRACTIONAL_ENERGY_CUTOFF,
        )
        weighted_acc = masked_weighted_accuracy(
            y_true=y,
            y_pred=predictions,
            energies=energy_weights,
            transform=wandb.config.accuracy_energy_weight_scheme,
            output_layer_segmentation_cutoff=OUTPUT_LAYER_SEGMENTATION_CUTOFF,
            fractional_energy_cutoff=FRACTIONAL_ENERGY_CUTOFF,
        )
    grads = tape.gradient(loss, model.trainable_variables)
    return loss, reg_acc, weighted_acc, grads


@tf.function
def val_step(x, y, energy_weights, model, loss_function):
    predictions = model(x, training=False)
    v_loss = masked_weighted_loss(
        y_true=y,
        y_pred=predictions,
        energies=energy_weights,
        transform=wandb.config.loss_energy_weight_scheme,
        loss_function=loss_function,
        fractional_energy_cutoff=FRACTIONAL_ENERGY_CUTOFF,
    )
    reg_acc = masked_regular_accuracy(
        y_true=y,
        y_pred=predictions,
        output_layer_segmentation_cutoff=OUTPUT_LAYER_SEGMENTATION_CUTOFF,
        fractional_energy_cutoff=FRACTIONAL_ENERGY_CUTOFF,
    )
    weighted_acc = masked_weighted_accuracy(
        y_true=y,
        y_pred=predictions,
        energies=energy_weights,
        transform=wandb.config.accuracy_energy_weight_scheme,
        output_layer_segmentation_cutoff=OUTPUT_LAYER_SEGMENTATION_CUTOFF,
        fractional_energy_cutoff=FRACTIONAL_ENERGY_CUTOFF,
    )
    return v_loss, reg_acc, weighted_acc, predictions


best_checkpoint_path = f"{MODELS_PATH}/PointNet_best.keras"
last_checkpoint_path = f"{MODELS_PATH}/PointNet_last_{EPOCHS=}.keras"


def main():
    tf.keras.backend.clear_session()
    seed = np.random.randint(0, 100)  # TF_SEED
    wandb.init(
        project="pointcloud",
        config={
            "dataset": EXPERIMENT_NAME,
            "split_seed": SPLIT_SEED,
            "tf_seed": seed,
            "delta_R": MAX_DISTANCE,
            "energy_scale": ENERGY_SCALE,
            "n_points_per_batch": MAX_SAMPLE_LENGTH,
            "batch_size": BATCH_SIZE,
            "n_epochs": EPOCHS,
            "early_stopping_patience": ES_PATIENCE,  # not used
            "output_activation": OUTPUT_ACTIVATION_FUNCTION,
            "accuracy_energy_weight_scheme": ACC_ENERGY_WEIGHTING,
            "loss_energy_weight_scheme": ACC_ENERGY_WEIGHTING,
            "min_hits_per_track": 25,
            "fractional_energy_cutoff": FRACTIONAL_ENERGY_CUTOFF,
            "lr_max": LR_MAX,
            "lr_min": LR_MIN,
            "lr_ramp_ep": LR_RAMP_EP,
            "lr_sus_ep": LR_SUS_EP,
            "lr_decay": LR_DECAY,
        },
        job_type="training",
        tags=["baseline"],
        # notes="This run reproduces Marko's setting. Consider this as the starting jet ML baseline.",
    )

    config = wandb.config

    train_steps = calculate_steps(TRAIN_DIR, BATCH_SIZE)  # 47
    val_steps = calculate_steps(VAL_DIR, BATCH_SIZE)  # 26
    print(f"{train_steps = };\t{val_steps = }")

    print(f"Setting training determinism based on {seed=}")
    set_global_determinism(seed=seed)

    model = PointNetSegmentation(
        MAX_SAMPLE_LENGTH,
        num_features=len(TRAIN_INPUTS),
        num_classes=1,
        output_activation_function=config.output_activation,
    )

    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum(
        [K.count_params(w) for w in model.non_trainable_weights]
    )

    print("Total params: {:,}".format(trainable_count + non_trainable_count))
    print("Trainable params: {:,}".format(trainable_count))
    print("Non-trainable params: {:,}".format(non_trainable_count))

    if __name__ == "__main__":
        optimizer = tf.keras.optimizers.Adam()

        train_loss_tracker = tf.metrics.Mean(name="train_loss")
        train_reg_acc = tf.metrics.Mean(name="train_regular_accuracy")
        train_weighted_acc = tf.metrics.Mean(name="train_weighted_accuracy")

        val_loss_tracker = tf.metrics.Mean(name="val_loss")
        val_reg_acc = tf.metrics.Mean(name="val_regular_accuracy")
        val_weighted_acc = tf.metrics.Mean(name="val_weighted_accuracy")
        recall_metric = tf.keras.metrics.Recall(
            thresholds=OUTPUT_LAYER_SEGMENTATION_CUTOFF
        )
        precision_metric = tf.keras.metrics.Precision(
            thresholds=OUTPUT_LAYER_SEGMENTATION_CUTOFF
        )

        loss_function = tf.keras.losses.BinaryCrossentropy(
            from_logits=False
        )  # NOTE: False for "sigmoid", True for "linear"
        val_f1_score = tf.keras.metrics.F1Score(
            threshold=OUTPUT_LAYER_SEGMENTATION_CUTOFF
        )

    # Callbacks
    # ModelCheckpoint
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_checkpoint_path,
        save_best_only=True,
        monitor="val_weighted_accuracy",  # Monitor validation loss
        mode="max",  # Save the model with the minimum validation loss
        save_weights_only=False,
        verbose=1,
    )
    checkpoint_callback.set_model(model)

    # EarlyStopping
    # early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    #     monitor="val_weighted_accuracy",  # Monitor validation loss
    #     mode="max",  # Trigger when validation loss stops decreasing
    #     patience=ES_PATIENCE,  # Number of epochs to wait before stopping if no improvement
    #     verbose=1,
    # )
    # early_stopping_callback.set_model(model)

    # Learning Rate Scheduler
    lr_callback = CustomLRScheduler(
        optim_lr=optimizer.learning_rate,
        lr_max=config.lr_max,  # 0.000015 * train_steps * BATCH_SIZE,
        lr_min=config.lr_min,  # 1e-7,
        lr_ramp_ep=config.lr_ramp_ep,  # 2,
        lr_sus_ep=config.lr_sus_ep,  # 0,
        lr_decay=config.lr_decay,  # 0.7,
        verbose=1,
    )


for epoch in range(EPOCHS):
    print(f"\nStart of epoch {epoch+1} of {EPOCHS}")
    start_time = time.time()

    # LR scheduler
    lr_callback.on_epoch_begin(epoch)

    train_loss_tracker.reset_state()
    train_reg_acc.reset_state()
    train_weighted_acc.reset_state()

    val_loss_tracker.reset_state()
    val_reg_acc.reset_state()
    val_weighted_acc.reset_state()

    val_f1_score.reset_state()
    recall_metric.reset_state()
    precision_metric.reset_state()

    val_true_labels = []
    val_predictions = []

    batch_loss_train, batch_accuracy_train, batch_weighted_accuracy_train = [], [], []
    for step, (x_batch_train, y_batch_train, e_weight_train) in enumerate(
        data_generator(TRAIN_DIR, BATCH_SIZE)
    ):
        x_batch_train = rfn.structured_to_unstructured(x_batch_train)
        if step >= train_steps:
            break
        loss_value, reg_acc_value, weighted_acc_value, grads = train_step(
            x_batch_train,
            y_batch_train,
            e_weight_train,
            model,
            optimizer,
            loss_function,
        )
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss_tracker.update_state(loss_value)
        train_reg_acc.update_state(reg_acc_value)
        train_weighted_acc.update_state(weighted_acc_value)

        print(
            f"\rEpoch {epoch + 1}, Step {step + 1}/{train_steps}, Training Loss: {train_loss_tracker.result().numpy():.4e}, Reg Acc: {train_reg_acc.result().numpy():.4f}, Weighted Acc: {train_weighted_acc.result().numpy():.4f}",
            end="",
        )
        batch_loss_train.append(train_loss_tracker.result().numpy())
        batch_accuracy_train.append(train_reg_acc.result().numpy())
        batch_weighted_accuracy_train.append(train_weighted_acc.result())

    print(f"\nTraining loss over epoch: {train_loss_tracker.result():.4e}")
    print(f"\nTime taken for training: {time.time() - start_time:.2f} sec")

    batch_loss_val, batch_accuracy_val, batch_weighted_accuracy_val = [], [], []
    for step, (x_batch_val, y_batch_val, e_weight_val) in enumerate(
        data_generator(VAL_DIR, BATCH_SIZE, False)
    ):
        x_batch_val = rfn.structured_to_unstructured(x_batch_val)
        if step >= val_steps:
            break
        val_loss_value, val_reg_acc_value, val_weighted_acc_value, predicted_y = (
            val_step(x_batch_val, y_batch_val, e_weight_val, model, loss_function)
        )
        val_loss_tracker.update_state(val_loss_value)
        val_reg_acc.update_state(val_reg_acc_value)
        val_weighted_acc.update_state(val_weighted_acc_value)

        mask = y_batch_val != -1  # remove non-energy points
        val_true_labels.extend(
            (y_batch_val[mask] > FRACTIONAL_ENERGY_CUTOFF).astype(np.float32)
        )
        val_predictions.extend(predicted_y.numpy()[mask])

        print(
            f"\rEpoch {epoch + 1}, Step {step + 1}/{val_steps}, Validation Loss: {val_loss_tracker.result().numpy():.4e}, Reg Acc: {val_reg_acc.result().numpy():.4f}, Weighted Acc: {val_weighted_acc.result().numpy():.4f}",
            end="",
        )

        batch_loss_val.append(val_loss_tracker.result().numpy())
        batch_accuracy_val.append(val_reg_acc.result().numpy())
        batch_weighted_accuracy_val.append(val_weighted_acc.result())

    val_true_labels = tf.convert_to_tensor(val_true_labels)
    val_predictions = tf.convert_to_tensor(val_predictions)

    val_f1_score.update_state(
        tf.expand_dims(val_true_labels, axis=-1),
        tf.expand_dims(val_predictions, axis=-1),
    )
    recall_metric.update_state(val_true_labels, val_predictions)
    precision_metric.update_state(val_true_labels, val_predictions)

    val_f1 = val_f1_score.result().numpy()
    precision = precision_metric.result().numpy()
    recall = recall_metric.result().numpy()

    print(f"\n Validation F1 Score: {val_f1}")
    print(f"\nValidation loss: {val_loss_tracker.result():.4e}")
    print(f"\nTime taken for validation: {time.time() - start_time:.2f} sec")

    wandb.log(
        {
            "epoch": epoch,
            "train/loss": train_loss_tracker.result().numpy(),
            "train/accuracy": train_reg_acc.result().numpy(),
            "train/weighted_accuracy": train_weighted_acc.result().numpy(),
            "val/loss": val_loss_tracker.result().numpy(),
            "val/accuracy": val_reg_acc.result().numpy(),
            "val/weighted_accuracy": val_weighted_acc.result().numpy(),
            "learning_rate": optimizer.learning_rate.numpy(),
            "val/f1_score": val_f1,
            "val/recall": recall,
            "val/precision": precision,
            # "gradients": [tf.reduce_mean(tf.abs(grad)).numpy() for grad in grads],
            # "weights": [
            #     tf.reduce_mean(tf.abs(weight)).numpy()
            #     for weight in model.trainable_variables
            # ],
        }
    )

    # callbacks
    # lr_callback.on_epoch_end(epoch)

    if epoch == 0:
        model.save(best_checkpoint_path)

    # discard first epochs to trigger callbacks
    if epoch > 5 & (val_weighted_acc.result().numpy() < 1):
        checkpoint_callback.on_epoch_end(
            epoch,
            logs={
                "val_loss": val_loss_tracker.result(),
                "val_weighted_accuracy": val_weighted_acc.result(),
            },
        )
        # early_stopping_callback.on_epoch_end(
        #     epoch,
        #     logs={
        #         "val_loss": val_loss_tracker.result(),
        #         "val_weighted_accuracy": val_weighted_acc.result(),
        #     },
        # )
        # if early_stopping_callback.stopped_epoch > 0:
        #     print(f"Early stopping triggered at epoch {epoch}")
        #     break


print("\n\nTraining completed!")

model.save(last_checkpoint_path)

# Log the best and last models to wandb
best_model_artifact = wandb.Artifact("best_baseline", type="model")
best_model_artifact.add_file(best_checkpoint_path)
wandb.log_artifact(best_model_artifact)

final_model_artifact = wandb.Artifact("last_epoch_baseline", type="model")
final_model_artifact.add_file(last_checkpoint_path)
wandb.log_artifact(final_model_artifact)

wandb.finish()


def run_sweep():
    sweep_configuration = {
        "method": "random",
        "metric": {"goal": "maximize", "name": "val/f1_score"},
        "parameters": {
            "loss_energy_weight_scheme": {
                "values": [
                    "absolute",
                    "square",
                    "normalize",
                    "standardize",
                    "threshold",
                ]
            },
            "lr_max": {"max": 0.2, "min": 0.001},
            "lr_min": {"max": 0.01, "min": 0.00001},
            "lr_ramp_ep": {"max": 10, "min": 1},
            "lr_sus_ep": {"max": 10, "min": 0},
            "lr_decay": {"max": 1.0, "min": 0.01},
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="pointcloud")
    if RUN_SWEEP:
        # os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID  # Set GPU
        print(f"Running Sweep: {sweep_id}")
        wandb.agent(sweep_id, function=main, count=200)
    else:
        # CUDA_VISIBLE_DEVICES=3 wandb agent 9bgjxxpe
        # CUDA_VISIBLE_DEVICES=3 wandb agent --project pointcloud --entity caloqvae 4zn0zn7q
        print(
            "RUN ONE (OR MORE) OF THE FOLLOWING:",
            *[f"CUDA_VISIBLE_DEVICES={i} wandb agent {sweep_id}" for i in range(6)],
            sep="\n",
        )


if __name__ == "__main__":
    if SWEEP:
        run_sweep()
    else:
        if __name__ == "__main__":
            # os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
            os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID  # Set GPU
        main()
