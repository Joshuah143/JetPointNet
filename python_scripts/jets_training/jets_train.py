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

REPO_PATH = Path.home() / "workspace/jetpointnet"
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
    MAX_SAMPLE_LENGTH,
    TRAIN,
    TRAIN_OUTPUT_DIRECTORY_NAME,
    TRAIN_DATASET_NAME,
    TRAIN_ALlOWED_SETS,
    TRAIN
)


# tf.config.run_functions_eagerly(True) - Useful when using the debugger - dont delete, but should not be used in production

# SET PATHS FOR I/O AND CONFIG
USER = Path.home().name
print(f"Logged in as {USER}")
if USER == "jhimmens":
    GPU_ID = "5"
    ASSIGN_GPU = True
elif USER == "luclissa":
    GPU_ID = "0"
    ASSIGN_GPU = False
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
else:
    raise Exception("UNKOWN USER")

if ASSIGN_GPU and __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

EXPERIMENT_NAME = f"{TRAIN_OUTPUT_DIRECTORY_NAME}/{TRAIN_DATASET_NAME}"
RESULTS_PATH = REPO_PATH / "result" / EXPERIMENT_NAME
RESULTS_PATH.mkdir(exist_ok=True, parents=True)
MODELS_PATH = REPO_PATH / "models" / EXPERIMENT_NAME
MODELS_PATH.mkdir(exist_ok=True, parents=True)

#REPLAY_SIMPLE_DS = Path("/home/jhimmens/workspace/jetpointnet/pnet_data/processed_files/progressive_training/delta/SavedNpz/deltaR=0.2_maxLen=650/energy_scale=1")
#REPLAY_SIMPLE_DS_NAME = 'delta'

#REPLAY_COMPLEX_DS = Path("/home/jhimmens/workspace/jetpointnet/pnet_data/processed_files/attempt_1_june_18/full_set/SavedNpz/deltaR=0.2_maxLen=650/energy_scale=1")
#REPLAY_COMPLEX_DS_NAME = 'dijet'

replay_model_path = Path("/home/jhimmens/workspace/jetpointnet/models/delta/progressive_training/PointNet_best_name=autumn-pyramid-836.keras")
tune_model_path = Path("/home/jhimmens/workspace/jetpointnet/models/rho/progressive_training/PointNet_best_name=dark-music-823.keras")

SIMPLE_SETS = []
COMPLEX_SETS = []

baseline_configuration = dict(
    SPLIT_SEED=62,
    MODEL_VERSION=1,
    INPUT_SETS=TRAIN_ALlOWED_SETS,
    EPOCH_COMPLEXITY=(EPOCH_COMPLEXITY := 1024*200),
    TF_SEED=np.random.randint(0, 100),
    MAX_SAMPLE_LENGTH=MAX_SAMPLE_LENGTH,  # 278 for delta R of 0.1, 859 for 0.2
    BATCH_SIZE=(BATCH_SIZE := 700),
    EPOCHS=150,
    IS_TUNE=False,
    REPLAY=(REPLAY := False),
    REPLAY_LINEAR_DECAY_RATE=0.02, # decrease of data from simple set
    REPLAY_MIN_FREQ=0.10, # steady state of simple set
    TRAIN_LR=0.04,
    SAVE_INTERMEDIATES=True,
    SAVE_FREQ=10, # Save intermediate models
    TUNE_LR=0.01,
    TRAIN_LR_DECAY=0.998,
    TUNE_LR_DECAY=0.99,
    LR_BETA1=0.98,
    LR_BETA2=0.999,
    ES_PATIENCE=15,
    ACC_ENERGY_WEIGHTING="square",
    LOSS_ENERGY_WEIGHTING="square",
    LOSS_FUNCTION="CategoricalCrossentropy",
    OUTPUT_ACTIVATION_FUNCTION="sigmoid",  # softmax, linear (requires changes to the BCE fucntion in the loss function)
    OUTPUT_LAYER_SEGMENTATION_CUTOFF=0.5,
    EARLY_STOPPING=False,
    TRAIN_STEPS=EPOCH_COMPLEXITY//BATCH_SIZE,
    VAL_STEPS=EPOCH_COMPLEXITY//BATCH_SIZE,
    # POTENTIALLY OVERWRITTEN BY THE WANDB SWEEP:
    # LR_MAX=0.000015,
    # LR_MIN=1e-5,
    # LR_RAMP_EP=2,
    # LR_SUS_EP=10,
    # LR_DECAY=0.7,
    METRIC="val/f1_score_focal",
    MODE="max",
)

# note that if you change the output activation function you must change the loss function
if (
    baseline_configuration["OUTPUT_ACTIVATION_FUNCTION"] in ["softmax", "sigmoid"]
    and baseline_configuration["OUTPUT_LAYER_SEGMENTATION_CUTOFF"] != 0.5
):
    raise Exception("Invalid OUTPUT_LAYER_SEGMENTATION_CUTOFF")
elif (
    baseline_configuration["OUTPUT_ACTIVATION_FUNCTION"] in ["linear"]
    and baseline_configuration["OUTPUT_LAYER_SEGMENTATION_CUTOFF"] != 0
):
    raise Exception("Invalid OUTPUT_LAYER_SEGMENTATION_CUTOFF")

# last_checkpoint_path = f"{MODELS_PATH}/PointNet_last_{epoch=}.keras"


"""
('event_number', np.int32),
('cell_ID', np.int32),
('track_ID', np.int32),
('delta_R', np.float32),

('truth_cell_focal_fraction_energy', np.float32),
('truth_cell_non_focal_fraction_energy', np.float32),
('truth_cell_neutral_fraction_energy', np.float32),
('truth_cell_total_energy', np.float32),

('category', np.int8),
('track_num', np.int32),
('normalized_x', np.float32),
('normalized_y', np.float32),
('normalized_z', np.float32),
('normalized_distance', np.float32),
('cell_sigma', np.float32),
('track_chi2_dof', np.float32),
("track_chi2_dof_cell_sigma", np.float32),
('cell_E', np.float32),
('track_pt', np.float32),
('track_pt_cell_E', np.float32),
"""

TRAIN_INPUTS = [
    "category",
    'delta_R',
    "track_num",
    "normalized_x",
    "normalized_y",
    "normalized_z",
    'track_chi2_dof',
    'cell_sigma',
    "normalized_distance",
    "cell_E",
    "track_pt",
]

TRAIN_TARGETS = [
    'truth_cell_focal_fraction_energy',
    'truth_cell_non_focal_fraction_energy',
    'truth_cell_neutral_fraction_energy',
]


def load_data_from_npz(npz_file):
    all_feats = np.load(npz_file)["feats"]
    feats = all_feats[:, :MAX_SAMPLE_LENGTH][TRAIN_INPUTS]  # discard tracking information
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
    npz_files = glob.glob(os.path.join(data_dir, set_name, "*.npz"))
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
                feats_buffer.extend(feats[last_batch_idx : last_index])
                targets_buffer.extend(
                    targets[last_batch_idx : last_index]
                )
                e_weights_buffer.extend(
                    e_weights[last_batch_idx : last_index]
                )

                # update unprocessed points and last index
                unprocessed_size -= fill_size
                last_batch_idx += fill_size

                # check if batch is full, in case yield + reset buffers
                if len(feats_buffer) == batch_size:
                    batch_feats, batch_targets, batch_e_weights = (
                        feats_buffer, targets_buffer, e_weights_buffer
                    )
                    feats_buffer, targets_buffer, e_weights_buffer = _init_buffers()
                    yield batch_feats, batch_targets, batch_e_weights

def consistant_data_generator(data_dir, data_sets: dict, batch_size: int, **kwargs):

    # Setup the genorators
    genorator_dict = {set_name: single_set_data_generator(data_dir, set_name, int(batch_size*inclusion_ratio)) for set_name, inclusion_ratio in data_sets.items()}
    while True:
        feats_buffer, targets_buffer, e_weights_buffer = _init_buffers()
        for generator in genorator_dict.values():
            feats_inner_buffer, targets_inner_buffer, e_weights_inner_buffer = next(generator)
            feats_buffer.extend(feats_inner_buffer)
            targets_buffer.extend(targets_inner_buffer)
            e_weights_buffer.extend(e_weights_inner_buffer)
        
        yield _format_batch(feats_buffer, targets_buffer, e_weights_buffer)


def progressive_data_generator(data_dir, simple_sets: list, complex_sets: list, batch_size: int, epoch: int, linear_decay: int, memory:int,  **kwargs):
    raise NotImplementedError("THIS FUNCTION MUST BE UPDATED TO USE THE CONISTANT GENERATOR BEFORE USE")
    percent_simple_data = max(1-epoch*linear_decay, memory)
    

    while True:
        for (simple_feats, simple_targets, simple_e_weights), (complex_feats, complex_targets, complex_e_weights) in zip(
                                            data_generator(data_dir,
                                                           simple_sets, 
                                                           int(batch_size*percent_simple_data),
                                                           **kwargs), 
                                            data_generator(data_dir, 
                                                           complex_sets, 
                                                           int(batch_size*(1-percent_simple_data)), 
                                                           **kwargs)):
            if simple_feats.size == 0:
                total_feats, total_targets, total_e_weights = complex_feats, complex_targets, complex_e_weights
            elif complex_feats.size == 0:
                total_feats, total_targets, total_e_weights = simple_feats, simple_targets, simple_e_weights
            else:
                total_feats = np.concatenate([simple_feats, complex_feats], axis=0)
                total_targets = np.concatenate([simple_targets, complex_targets], axis=0)
                total_e_weights = np.concatenate([simple_e_weights, complex_e_weights], axis=0)
            
            yield total_feats, total_targets, total_e_weights


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
    fine_tune: bool = False,
    replay: bool = False
):
    model = PointNetSegmentation(
        num_points=num_points,
        num_features=num_features,
        num_classes=num_classes,
        output_activation_function=output_activation,
        model_version=model_version
    )

    if fine_tune and replay:
        raise Exception('CANNOT FINE TUNE AND REPLAY SIMULTANEOUSLY')

    if fine_tune:
        model.load_weights(tune_model_path)

        model.trainable = True

       # print(model.summary()) - we are ok to set roughly last 12 as trainable in a tune situation

        for layer in model.layers[:-4]:
            layer.trainable = False

    if replay:
         model.load_weights(replay_model_path)

    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum(
        [K.count_params(w) for w in model.non_trainable_weights]
    )

    print("Total params: {:,}".format(trainable_count + non_trainable_count))
    print("Trainable params: {:,}".format(trainable_count))
    print("Non-trainable params: {:,}".format(non_trainable_count))
    return model, trainable_count


# if USE_WANDB:
#     wandb.init(
#         project="pointcloud",
#         config={
#             "dataset": EXPERIMENT_NAME,
#             "split_seed": SPLIT_SEED,
#             "tf_seed": seed,
#             "delta_R": MAX_DISTANCE,
#             "energy_scale": ENERGY_SCALE,
#             "n_points_per_batch": MAX_SAMPLE_LENGTH,
#             "batch_size": BATCH_SIZE,
#             "n_epochs": EPOCHS,
#             "learning_rate": LR,
#             "early_stopping_patience": ES_PATIENCE,
#             "output_activation": "softmax",
#             "detlaR": MAX_DISTANCE,
#             "min_hits_per_track": 25,
#         },
#         job_type="training",
#         tags=["baseline"],
#         notes="This run reproduces Marko's setting. Consider this as the starting jet ML baseline.",
#     )

def merge_configurations(priority_config, baseline_config):
    for hyperparam, value in priority_config.items():
        if hyperparam in baseline_config.keys():
            baseline_config[hyperparam] = {"value": value}
        else:
            raise AttributeError(f"{hyperparam} set in experimental config, but not found in baseline config, this parameter is not used and is likely set by error. Please check the config is in `baseline_config`.")
    return baseline_config


def train(experimental_configuration: dict = {}):
    run_config = merge_configurations(experimental_configuration, baseline_configuration)
    with wandb.init(
        project="pointcloud", 
        config=run_config, 
        job_type="training", 
        tags=[TRAIN_OUTPUT_DIRECTORY_NAME, 
              TRAIN_DATASET_NAME, 
              str(TRAIN_ALlOWED_SETS.keys())] if not REPLAY else [TRAIN_OUTPUT_DIRECTORY_NAME, 
                                                      TRAIN_DATASET_NAME, 
                                                      f"from={SIMPLE_SETS}", f"to={COMPLEX_SETS}"],
        notes=""
    ) as run:
        config = wandb.config

        # number of steps and seed
        train_steps = config.TRAIN_STEPS # calculate_steps(TRAIN_DIR, config.BATCH_SIZE)  # 47
        val_steps = config.VAL_STEPS # calculate_steps(VAL_DIR, config.BATCH_SIZE)  # 26
        print(f"{train_steps = };\t{val_steps = }")

        seed = config.TF_SEED
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
                    transform=config.LOSS_ENERGY_WEIGHTING,
                )
                reg_acc = masked_regular_accuracy(
                    y_true=y,
                    y_pred=predictions,
                    x_class=x_class,
                )
                weighted_acc = masked_weighted_accuracy(
                    y_true=y,
                    y_pred=predictions,
                    energies=energy_weights,
                    x_class=x_class,
                    transform=config.ACC_ENERGY_WEIGHTING,
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
                transform=config.LOSS_ENERGY_WEIGHTING,
                x_class=x_class,
                loss_function=loss_function,
            )
            reg_acc = masked_regular_accuracy(
                y_true=y,
                y_pred=predictions,
                x_class=x_class,
            )
            weighted_acc = masked_weighted_accuracy(
                y_true=y,
                y_pred=predictions,
                energies=energy_weights,
                x_class=x_class,
                transform=config.ACC_ENERGY_WEIGHTING,
            )
            return v_loss, reg_acc, weighted_acc, predictions

        # model, trackers and callbacks and setup
        model, trainable_params = _setup_model(
            num_points=config.MAX_SAMPLE_LENGTH,
            num_features=len(TRAIN_INPUTS),
            num_classes=len(TRAIN_TARGETS),
            output_activation=config.OUTPUT_ACTIVATION_FUNCTION,
            fine_tune=config.IS_TUNE,
            replay=config.REPLAY,
            model_version=config.MODEL_VERSION
        )

        wandb.log({"trainable_params": trainable_params})

        train_loss_tracker = tf.metrics.Mean(name="train_loss")
        train_reg_acc = tf.metrics.Mean(name="train_regular_accuracy")
        train_weighted_acc = tf.metrics.Mean(name="train_weighted_accuracy")

        val_loss_tracker = tf.metrics.Mean(name="val_loss")
        val_reg_acc = tf.metrics.Mean(name="val_regular_accuracy")
        val_weighted_acc = tf.metrics.Mean(name="val_weighted_accuracy")
        mean_iou_metric = tf.keras.metrics.OneHotMeanIoU(len(TRAIN_TARGETS))
        val_f1_score = tf.keras.metrics.F1Score(
            threshold=config.OUTPUT_LAYER_SEGMENTATION_CUTOFF
        )

        # Callbacks
        # ModelCheckpoint
        best_checkpoint_path = f"{MODELS_PATH}/PointNet_best_name={run.name}.keras"
        
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=best_checkpoint_path,
            save_best_only=True,
            monitor=config.METRIC,  # Monitor validation loss
            mode=config.MODE,  # Save the model with the minimum validation loss
            save_weights_only=False,
            verbose=1,
        )
        checkpoint_callback.set_model(model)

        # EarlyStopping
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor=config.METRIC,  # "val_weighted_accuracy",  # Monitor validation loss
            mode=config.MODE,  # "max",  # Trigger when validation loss stops decreasing
            patience=config.ES_PATIENCE,  # Number of epochs to wait before stopping if no improvement
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
            initial_learning_rate=config.TRAIN_LR if not config.IS_TUNE else config.TUNE_LR,
            decay_steps=train_steps,
            decay_rate=config.TRAIN_LR_DECAY if not config.IS_TUNE else config.TUNE_LR_DECAY,
        )

        # Optimizer & Loss
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=config.LR_BETA1,
            beta_2=config.LR_BETA2,
            # decay=config.LR_DECAY,
        )

        
        # Will raise AttributeError if the loss function is not found
        logits = config.OUTPUT_ACTIVATION_FUNCTION == "linear"
        loss_function = getattr(tf.keras.losses, config.LOSS_FUNCTION)(
            from_logits=logits, # NOTE: False for "sigmoid", True for "linear"
            # reduction='none',
        )
            
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

        for epoch in range(config.EPOCHS):
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

            val_f1_score.reset_state()
            mean_iou_metric.reset_state()

            val_true_labels = []
            val_predictions = []

            # train step
            batch_loss_train, batch_accuracy_train, batch_weighted_accuracy_train = (
                [],
                [],
                [],
            )
            if config.REPLAY:
                _train_generator = progressive_data_generator(NPZ_SAVE_LOC(TRAIN) / 'train',
                                                            SIMPLE_SETS,
                                                            COMPLEX_SETS,
                                                            config.BATCH_SIZE,
                                                            epoch,
                                                            config.REPLAY_LINEAR_DECAY_RATE,
                                                            config.REPLAY_MIN_FREQ)
            else:
                _train_generator = consistant_data_generator(NPZ_SAVE_LOC(TRAIN) / 'train', 
                                                config.INPUT_SETS,
                                                config.BATCH_SIZE)
            train_generator = enumerate(_train_generator)

            for step, (x_batch_train_named, y_batch_train, e_weight_train) in train_generator:
                x_catagories = x_batch_train_named['category']
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
                    x_catagories
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

            if config.REPLAY:
                _val_generator = progressive_data_generator(NPZ_SAVE_LOC(TRAIN) / 'val',
                                                            SIMPLE_SETS,
                                                            COMPLEX_SETS,
                                                            config.BATCH_SIZE,
                                                            epoch,
                                                            config.REPLAY_LINEAR_DECAY_RATE,
                                                            config.REPLAY_MIN_FREQ)
            else:
                _val_generator = consistant_data_generator(NPZ_SAVE_LOC(TRAIN) / 'val', 
                                                config.INPUT_SETS,
                                                config.BATCH_SIZE)
            val_generator = enumerate(_val_generator)
            
            for step, (x_batch_val_named, y_batch_val_named, e_weight_val) in val_generator:
                x_catagories_val = x_batch_val_named['category']
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
                    x_batch_val, y_batch_val, e_weight_val, model, loss_function, x_catagories_val
                )
                val_loss_tracker.update_state(val_loss_value)
                val_reg_acc.update_state(val_reg_acc_value)
                val_weighted_acc.update_state(val_weighted_acc_value)

                mask = x_batch_val_named['category'] == 1   # remove non-energy points
                val_true_labels.extend(y_batch_val[mask])
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

            val_true_labels = tf.cast(val_true_labels, dtype=tf.float32)
            val_predictions = tf.cast(val_predictions, dtype=tf.float32)

            val_f1_score.update_state(
                val_true_labels, # tf.expand_dims(val_true_labels, axis=-1),
                val_predictions # tf.expand_dims(val_predictions, axis=-1),
            )
            mean_iou_metric.update_state(val_true_labels, val_predictions)

            val_f1 = val_f1_score.result().numpy()

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
                    "val/f1_score_focal": val_f1_score.result().numpy()[0],
                    "val/f1_score_non_focal": val_f1_score.result().numpy()[1],
                    "val/f1_score_neutral": val_f1_score.result().numpy()[2],
                    "val/mean_iou": mean_iou_metric.result().numpy(),
                    "percent_simple_data": max(1-epoch*config.REPLAY_LINEAR_DECAY_RATE, config.REPLAY_MIN_FREQ) if config.REPLAY else 1,
                    # "gradients": [tf.reduce_mean(tf.abs(grad)).numpy() for grad in grads],
                    # "weights": [
                    #     tf.reduce_mean(tf.abs(weight)).numpy()
                    #     for weight in model.trainable_variables
                    # ],
                }
            )

            # callbacks

            # discard first epochs to trigger callbacks
            if epoch > 5:

                if config.SAVE_INTERMEDIATES and epoch % config.SAVE_FREQ == 0:
                    checkpoint_path = f"{MODELS_PATH}/PointNet_{epoch=}_name={run.name}.keras"
                    model.save(checkpoint_path)

                checkpoint_callback.on_epoch_end(
                    epoch,
                    # TODO: adapt for user-defined metric tracking
                    logs={
                        "val_loss": val_loss_tracker.result(),
                        "val/accuracy": val_reg_acc.result(),
                        "val_weighted_accuracy": val_weighted_acc.result(),
                        "val/f1_score_focal": val_f1_score.result().numpy()[0],
                        "val/f1_score_non_focal": val_f1_score.result().numpy()[1],
                        "val/f1_score_neutral": val_f1_score.result().numpy()[2],
                        "val/mean_iou": mean_iou_metric.result().numpy(),
                    },
                )
                if config.EARLY_STOPPING:
                    early_stopping_callback.on_epoch_end(
                        epoch,
                        logs={
                            "val_loss": val_loss_tracker.result(),
                            "val/accuracy": val_reg_acc.result(),
                            "val_weighted_accuracy": val_weighted_acc.result(),
                            "val/f1_score_focal": val_f1_score.result().numpy()[0],
                            "val/f1_score_non_focal": val_f1_score.result().numpy()[1],
                            "val/f1_score_neutral": val_f1_score.result().numpy()[2],
                            "val/mean_iou": mean_iou_metric.result().numpy(),
                           
                        },
                    )
                    if early_stopping_callback.model.stop_training:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break
        print("\n\nTraining completed!")

        last_checkpoint_path = f"{MODELS_PATH}/PointNet_last_{epoch=}_name={run.name}.keras"
        model.save(last_checkpoint_path)

        # Log the best and last models to wandb

        best_model_artifact = wandb.Artifact("best_baseline", type="model")
        best_model_artifact.add_file(best_checkpoint_path)
        wandb.log_artifact(best_model_artifact)

        final_model_artifact = wandb.Artifact("last_epoch_baseline", type="model")
        final_model_artifact.add_file(last_checkpoint_path)
        wandb.log_artifact(final_model_artifact)


if __name__ == "__main__":
    train({})
