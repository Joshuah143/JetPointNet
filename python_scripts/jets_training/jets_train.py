import sys
from pathlib import Path

# CERNBOX = os.environ["CERNBOX"]
REPO_PATH = Path.home() / "workspace/jetpointnet"
# SCRIPT_PATH = REPO_PATH / "python_scripts/data_processing/jets"
SCRIPT_PATH = REPO_PATH / "python_scripts"
sys.path.append(str(SCRIPT_PATH))


import numpy as np
import tensorflow as tf
import os
import glob
import math
import time
import wandb
from tqdm.auto import tqdm
from jets_training.models.JetPointNet import (
    PointNetSegmentation,
    masked_weighted_bce_loss,
    masked_regular_accuracy,
    masked_weighted_accuracy,
)
from data_processing.jets.preprocessing_header import MAX_DISTANCE

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set GPU

# SET PATHS FOR I/O AND CONFIG
OUTPUT_DIRECTORY_NAME = "ttbar"
DATASET_NAME = "benchmark"
ENERGY_SCALE = 1000
EXPERIMENT_NAME = f"{OUTPUT_DIRECTORY_NAME}/{DATASET_NAME}"
RESULTS_PATH = REPO_PATH / "result" / EXPERIMENT_NAME
RESULTS_PATH.mkdir(exist_ok=True, parents=True)
MODELS_PATH = REPO_PATH / "models" / EXPERIMENT_NAME
MODELS_PATH.mkdir(exist_ok=True, parents=True)

NPZ_SAVE_LOC = (
    REPO_PATH
    / "pnet_data/processed_files"
    / OUTPUT_DIRECTORY_NAME
    / DATASET_NAME
    / "SavedNpz"
    / f"deltaR={MAX_DISTANCE}"
    / f"energy_scale={ENERGY_SCALE}"
)

SPLIT_SEED = 62
MAX_SAMPLE_LENGTH = 278
BATCH_SIZE = 480
EPOCHS = 10
LR = 0.001
TRAIN_DIR = NPZ_SAVE_LOC / "train"
VAL_DIR = NPZ_SAVE_LOC / "val"


def load_data_from_npz(npz_file):
    data = np.load(npz_file)
    feats = data["feats"][:, :MAX_SAMPLE_LENGTH, 1:]  # discard eventNumber
    frac_labels = data["frac_labels"][:, :MAX_SAMPLE_LENGTH]
    energy_weights = data["tot_truth_e"][:, :MAX_SAMPLE_LENGTH]
    return feats, frac_labels, energy_weights


def data_generator(data_dir, batch_size, drop_last=True):
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    while True:
        np.random.shuffle(npz_files)
        for npz_file in npz_files:
            feats, frac_labels, e_weights = load_data_from_npz(npz_file)
            dataset_size = feats.shape[0]
            for i in range(0, dataset_size, batch_size):
                end_index = i + batch_size
                if end_index > dataset_size:
                    if drop_last:
                        continue  # Drop last smaller batch
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


train_steps = calculate_steps(TRAIN_DIR, BATCH_SIZE)  # 47
val_steps = calculate_steps(VAL_DIR, BATCH_SIZE)  # 26
print(f"{train_steps = };\t{val_steps = }")


wandb.init(
    project="pointcloud",
    config={
        "dataset": EXPERIMENT_NAME,
        "seed": SPLIT_SEED,
        "delta_R": MAX_DISTANCE,
        "energy_scale": ENERGY_SCALE,
        "n_points_per_batch": MAX_SAMPLE_LENGTH,
        "batch_size": BATCH_SIZE,
        "n_epochs": EPOCHS,
        "learning_rate": LR,
    },
    job_type="training",
)

model = PointNetSegmentation(MAX_SAMPLE_LENGTH, 1)
optimizer = tf.keras.optimizers.Adam(learning_rate=(LR))


@tf.function
def train_step(x, y, energy_weights, model, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = masked_weighted_bce_loss(y, predictions, energy_weights)
        reg_acc = masked_regular_accuracy(y, predictions, energy_weights)
        weighted_acc = masked_weighted_accuracy(y, predictions, energy_weights)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, reg_acc, weighted_acc, grads


@tf.function
def val_step(x, y, energy_weights, model):
    predictions = model(x, training=False)
    v_loss = masked_weighted_bce_loss(y, predictions, energy_weights)
    reg_acc = masked_regular_accuracy(y, predictions, energy_weights)
    weighted_acc = masked_weighted_accuracy(y, predictions, energy_weights)
    return v_loss, reg_acc, weighted_acc


train_loss_tracker = tf.metrics.Mean(name="train_loss")
val_loss_tracker = tf.metrics.Mean(name="val_loss")
train_reg_acc = tf.metrics.Mean(name="train_regular_accuracy")
train_weighted_acc = tf.metrics.Mean(name="train_weighted_accuracy")
val_reg_acc = tf.metrics.Mean(name="val_regular_accuracy")
val_weighted_acc = tf.metrics.Mean(name="val_weighted_accuracy")

# Setup ModelCheckpoint callback
checkpoint_path = f"{MODELS_PATH}/PointNet_best.keras"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    monitor="val_loss",  # Monitor validation loss
    mode="min",  # Save the model with the minimum validation loss
    save_weights_only=False,
    verbose=1,
)
checkpoint_callback.set_model(model)


for epoch in range(EPOCHS):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    train_loss_tracker.reset_state()
    val_loss_tracker.reset_state()
    train_reg_acc.reset_state()
    train_weighted_acc.reset_state()
    val_reg_acc.reset_state()
    val_weighted_acc.reset_state()

    batch_loss_train, batch_accuracy_train, batch_weighted_accuracy_train = [], [], []
    for step, (x_batch_train, y_batch_train, e_weight_train) in enumerate(
        data_generator(TRAIN_DIR, BATCH_SIZE)
    ):
        if step >= train_steps:
            break
        loss_value, reg_acc_value, weighted_acc_value, grads = train_step(
            x_batch_train, y_batch_train, e_weight_train, model, optimizer
        )
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

    print(f"\nTraining loss over epoch: {train_loss_tracker.result():.4f}")
    print(f"Time taken for training: {time.time() - start_time:.2f} sec")

    batch_loss_val, batch_accuracy_val, batch_weighted_accuracy_val = [], [], []
    for step, (x_batch_val, y_batch_val, e_weight_val) in enumerate(
        data_generator(VAL_DIR, BATCH_SIZE)
    ):
        if step >= val_steps:
            break
        val_loss_value, val_reg_acc_value, val_weighted_acc_value = val_step(
            x_batch_val, y_batch_val, e_weight_train, model
        )
        val_loss_tracker.update_state(val_loss_value)
        val_reg_acc.update_state(val_reg_acc_value)
        val_weighted_acc.update_state(val_weighted_acc_value)
        print(
            f"\rEpoch {epoch + 1}, Step {step + 1}/{val_steps}, Validation Loss: {val_loss_tracker.result().numpy():.4e}, Reg Acc: {val_reg_acc.result().numpy():.4f}, Weighted Acc: {val_weighted_acc.result().numpy():.4f}",
            end="",
        )
        batch_loss_val.append(val_loss_tracker.result().numpy())
        batch_accuracy_val.append(val_reg_acc.result().numpy())
        batch_weighted_accuracy_val.append(val_weighted_acc.result())

    print(f"Validation loss: {val_loss_tracker.result():.4e}")
    print(f"Time taken for validation: {time.time() - start_time:.2f} sec")

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
            "gradients": [tf.reduce_mean(tf.abs(grad)).numpy() for grad in grads],
            "weights": [
                tf.reduce_mean(tf.abs(weight)).numpy()
                for weight in model.trainable_variables
            ],
        }
    )
    model.save(f"{MODELS_PATH}/PointNet_last_{epoch=}.h5")
    # print("Model saved.")
    # Checkpointing the best model based on validation loss
    checkpoint_callback.on_epoch_end(
        epoch, logs={"val_loss": val_loss_tracker.result()}
    )


print("Training completed!")

model.save(f"{MODELS_PATH}/PointNet_last_{epoch=}.keras")
wandb.finish()
