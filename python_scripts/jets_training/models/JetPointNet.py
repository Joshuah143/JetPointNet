"""
Adapted From
https://github.com/lattice-ai/pointnet/tree/master

Original Architecture From Pointnet Paper:
https://arxiv.org/pdf/1612.00593.pdf
"""

import tensorflow as tf
import numpy as np
import keras
import os
import random
import sys
from pathlib import Path
import os

REPO_PATH = Path.home() / "workspace/jetpointnet"
SCRIPT_PATH = REPO_PATH / "python_scripts"
sys.path.append(str(SCRIPT_PATH))

from data_processing.jets.preprocessing_header import POINT_TYPE_ENCODING


# =======================================================================================================================
# ============ Weird Stuff ==============================================================================================


TF_SEED = 2


def _set_seeds(seed: int = TF_SEED):
    """
    Initialize seeds for all libraries which might have stochastic behavior
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed: int = TF_SEED):
    """
    Activate Tensorflow deterministic behavior
    """
    _set_seeds(seed=seed)

    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


class SaveModel(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.model.save("JetPointNet_{epoch}.hd5".format(epoch))


class CustomMaskingLayer(tf.keras.layers.Layer):
    # For masking out the inputs properly, based on points for which the last value in the point's array (it's "type") is "-1"
    def __init__(self, **kwargs):
        super(CustomMaskingLayer, self).__init__(**kwargs)

    def call(self, inputs):
        mask = tf.not_equal(inputs[:, :, -1], 1)  # Masking
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, -1)
        return inputs * mask

    def compute_output_shape(self, input_shape):
        return input_shape


class OrthogonalRegularizer(tf.keras.regularizers.OrthogonalRegularizer):
    # Used in Tnet in PointNet for transforming everything to same space
    def __init__(self, num_features=9, l2=0.001):
        self.num_features = num_features
        self.l2 = l2
        self.I = tf.eye(num_features)

    def __call__(self, inputs):
        A = tf.reshape(inputs, (-1, self.num_features, self.num_features))
        AAT = tf.tensordot(A, A, axes=(2, 2))
        AAT = tf.reshape(AAT, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2 * tf.square(AAT - self.I))

    def get_config(self):
        # Return a dictionary containing the parameters of the regularizer to allow for model serialization
        return {"num_features": self.num_features, "l2": self.l2}


"""
def rectified_TSSR_Activation(x):
    a = 0.01  # leaky ReLu style slope when negative
    b = 0.1  # sqrt(x) damping coefficient when x > 1

    # Adapted from https://arxiv.org/pdf/2308.04832.pdf
    # An activation function that's linear when 0 < x < 1 and (an adjusted) sqrt when x > 1,
    # behaves like leaky ReLU when x < 0.

    # 'a' is the slope coefficient for x < 0.
    # 'b' is the value to multiply by the sqrt(x) part.

    negative_condition = x < 0
    small_positive_condition = tf.logical_and(tf.greater_equal(x, 0), tf.less(x, 1))
    # large_positive_condition = x >= 1

    negative_part = a * x
    small_positive_part = x
    large_positive_part = tf.sign(x) * (b * tf.sqrt(tf.abs(x)) - b + 1)

    return tf.where(
        negative_condition,
        negative_part,
        tf.where(small_positive_condition, small_positive_part, large_positive_part),
    )
"""


# Never used
def custom_sigmoid(x, a=3.0):
    return 1 / (1 + tf.exp(-a * x))


# Never used
def hard_sigmoid(x):
    return tf.keras.backend.cast(x > 0, dtype=tf.float32)


# =======================================================================================================================
# =======================================================================================================================


# =======================================================================================================================
# ============ Main Model Blocks ========================================================================================


def conv_mlp(input_tensor, filters, dropout_rate=None, apply_attention=False, name=None):
    if name is not None:
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, activation="relu", name=name)(
            input_tensor
        )
    else: 
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, activation="relu")(
            input_tensor
        )
    x = tf.keras.layers.BatchNormalization()(x)

    if apply_attention:
        # Self-attention
        attention_output_self = tf.keras.layers.MultiHeadAttention(
            num_heads=2, key_dim=filters
        )(x, x)
        attention_output_self = tf.keras.layers.LayerNormalization()(
            attention_output_self + x
        )

        # Cross-attention
        attention_output_cross = tf.keras.layers.MultiHeadAttention(
            num_heads=2, key_dim=filters
        )(attention_output_self, x)
        attention_output_cross = tf.keras.layers.LayerNormalization()(
            attention_output_cross + attention_output_self
        )

        x = attention_output_cross

    if dropout_rate is not None:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    return x


def dense_block(input_tensor, units, dropout_rate=None, regularizer=None):
    x = tf.keras.layers.Dense(units, kernel_regularizer=regularizer)(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    if dropout_rate is not None:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


def TNet(input_tensor, size, add_regularization=False): # JH: why dont we add redularization? Not an issue I dont think, but super strange
    # size is either 6 for the first TNet or 64 for the second
    x = conv_mlp(input_tensor, 64)
    x = conv_mlp(x, 128)
    x = conv_mlp(x, 1024)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = dense_block(x, 512)
    x = dense_block(x, 256)
    if add_regularization:
        reg = OrthogonalRegularizer(size)
    else:
        reg = None
    x = dense_block(x, size * size, regularizer=reg)
    x = tf.keras.layers.Reshape((size, size))(x)
    return x


def PointNetSegmentation(
    num_points, num_features, num_classes, output_activation_function, model_version
):
    input_points = tf.keras.Input(shape=(num_points, num_features))

    # Masking layer to ignore points with the last feature index as -1
    masks = tf.keras.layers.Lambda(
        lambda x: tf.not_equal(x[:, :, -1], 1), output_shape=(num_points,)
    )(input_points)
    masks = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(
        masks
    )  # Cast boolean to float for multiplication
    masks = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1))(
        masks
    )  # Expand dimensions to apply mask

    # Apply mask
    input_points_masked = tf.keras.layers.Multiply()([input_points, masks])

    # energy = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, :, 4], -1), name='e')(input_points_masked)

    # T-Net for input transformation
    input_tnet = TNet(
        input_points_masked, num_features
    )  # Assuming TNet is properly defined elsewhere
    x = tf.keras.layers.Dot(axes=(2, 1))([input_points_masked, input_tnet])
    x = conv_mlp(x, 96)  # JH: This should be 64
    x = conv_mlp(x, 96)  # JH: This should be 64
    point_features = x

    # T-Net for feature transformation
    feature_tnet = TNet(x, 96, add_regularization=True)
    x = tf.keras.layers.Dot(axes=(2, 1))([x, feature_tnet])
    x = conv_mlp(x, 128) # JH: this should be 64?
    x = conv_mlp(x, 256) # JG: this should be 128?
    x = conv_mlp(x, 1024)

    # Get global features and expand
    global_feature = tf.keras.layers.GlobalMaxPooling1D(name="GlobalPooling")(x)
    global_feature_expanded = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 1))(
        global_feature
    )
    global_feature_expanded = tf.keras.layers.Lambda(
        lambda x: tf.tile(x, [1, num_points, 1])
    )(global_feature_expanded)

    # Segmentaion head
    if model_version == 0: # ~5M params
        c = tf.keras.layers.Concatenate()([point_features, global_feature_expanded])

        c = conv_mlp(c, 512, apply_attention=False)
        c = conv_mlp(c, 256, apply_attention=False)

        c = conv_mlp(c, 128, dropout_rate=0.3)
    elif model_version == 1: # ~6M params
        c = tf.keras.layers.Concatenate()([point_features, global_feature_expanded])

        c = conv_mlp(c, 1024, apply_attention=False)

        c = conv_mlp(c, 512, apply_attention=False)
        c = conv_mlp(c, 256, apply_attention=False)
        c = conv_mlp(c, 128, apply_attention=False)

        c = conv_mlp(c, 128, dropout_rate=0.3)
    elif model_version == 2: # ~7M params
        c = tf.keras.layers.Concatenate()([point_features, global_feature_expanded])

        c = conv_mlp(c, 1024, apply_attention=False)
       
        c = conv_mlp(c, 1024, apply_attention=False)
        c = conv_mlp(c, 512, apply_attention=False)
        c = conv_mlp(c, 256, apply_attention=False)

        c = conv_mlp(c, 256, dropout_rate=0.3)
    elif model_version == 3: 
        c = tf.keras.layers.Concatenate()([point_features, global_feature_expanded])

        c = conv_mlp(c, 2048, apply_attention=False)
        c = conv_mlp(c, 1024, apply_attention=False)
        c = conv_mlp(c, 1024, apply_attention=False)
        c = conv_mlp(c, 512, apply_attention=False)
        c = conv_mlp(c, 512, apply_attention=False)

        c = conv_mlp(c, 256, dropout_rate=0.3)
    else:
        raise Exception("INVALID MODEL VERSION")


    segmentation_output = tf.keras.layers.Conv1D(
        num_classes, kernel_size=1, activation=output_activation_function, name="SEG"
    )(c)

    model = tf.keras.Model(inputs=input_points, outputs=segmentation_output)

    return model


# =======================================================================================================================
# =======================================================================================================================


# =======================================================================================================================
# ============ Losses ===================================================================================================


@tf.autograph.experimental.do_not_convert
def masked_weighted_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    energies: tf.Tensor,
    loss_function: tf.keras.losses.Loss,
    x_class: tf.Tensor,
    transform: None | str = None,
    energy_threshold: float = 0,
):
    """
    Computes the masked weighted loss of predictions.

    Parameters:
    y_true (tf.Tensor): True labels.
    y_pred (tf.Tensor): Predicted labels.
    energies (tf.Tensor): Weights for each prediction.
    loss_function: (tf.keras.losses.Loss): The loss function to call with the model outputs
    x_class: (tf.Tensor): The point-type for each cell, CELL, PAD, TRACK, etc
    transform (str, optional): Transformation to apply to energies. Possible values:
        - None: no transformation (default).
        - "absolute": absolute value.
        - "square": square.
        - "normalize": batch-normalize to zero mean and unit variance.
        - "standardize": batch-standardize to zero mean and unit variance.
        - "threshold": threshold at 0 --> discard contributions by negative energies.
    energy_threshold (float, optional): the threshold to cutoff energy weighting if "threshold" is the transform

    Returns:
    tf.Tensor: standardized loss.
    """

    # Transform energy weights
    match transform:
        case "absolute":
            energies = tf.abs(energies)
        case "square":
            energies = tf.square(energies)
        case "normalize":
            energies = (energies - tf.reduce_min(energies)) / (
                tf.reduce_max(energies) - tf.reduce_min(energies) + 1e-5
            )
        case "standardize":
            energies = (energies - tf.reduce_mean(energies)) / (
                tf.math.reduce_std(energies) + 1e-5
            )
        case "threshold":
            energies = tf.cast(tf.greater(energies, energy_threshold), tf.float32)
        case None | "none":
            pass
        case _:
            raise ValueError(f"Unknown transform value: {transform}")
        

    valid_mask = tf.equal(x_class, POINT_TYPE_ENCODING['cell']) 
    valid_mask = tf.cast(valid_mask, tf.float32)

    energies_times_mask = energies * valid_mask

    # Calculate categorical cross-entropy loss
    weighted_loss = loss_function(y_true, y_pred, sample_weight=energies_times_mask)

    return weighted_loss


@tf.autograph.experimental.do_not_convert
def masked_weighted_accuracy(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    energies: tf.Tensor,
    x_class: tf.Tensor,
    unweighted_accuracy_metric: tf.keras.metrics.Metric,
    weighted_accuracy_metric: tf.keras.metrics.Metric,
    transform: None | str = None,
    energy_threshold: float = 0, 
):
    """
    Computes the masked weighted accuracy of predictions.

    Parameters:
    y_true (tf.Tensor): True labels.
    y_pred (tf.Tensor): Predicted labels.
    energies (tf.Tensor): Weights for each prediction.
    x_class: (tf.Tensor): The point-type for each cell, CELL, PAD, TRACK, etc
    unweighted_accuracy_metric (tf.keras.metrics.Metric): The metric to be called with outputs
    weighted_accuracy_metric (tf.keras.metrics.Metric): The metric to be called with outputs
    transform (str, optional): Transformation to apply to energies. Possible values:
        - None: no transformation (default).
        - "absolute": absolute value.
        - "square": square.
        - "normalize": batch-normalize to zero mean and unit variance.
        - "standardize": batch-standardize to zero mean and unit variance.
        - "threshold": threshold at 0 --> discard contributions by negative energies.
    energy_threshold (float, optional): the threshold to cutoff energy weighting if "threshold" is the transform
    """
    # Transform energy weights
    match transform:
        case "absolute":
            energies = tf.abs(energies)
        case "square":
            energies = tf.square(energies)
        case "normalize":
            energies = (energies - tf.reduce_min(energies)) / (
                tf.reduce_max(energies) - tf.reduce_min(energies) + 1e-5
            )
        case "standardize":
            energies = (energies - tf.reduce_mean(energies)) / (
                tf.math.reduce_std(energies) + 1e-5
            )
        case "threshold":
            energies = tf.cast(tf.greater(energies, energy_threshold), tf.float32)
        case None | "none":
            pass
        case _:
            raise ValueError(f"Unknown transform value: {transform}")

    valid_mask = tf.equal(x_class, POINT_TYPE_ENCODING['cell']) 
    valid_mask = tf.cast(valid_mask, tf.float32)

    energies_times_mask = energies * valid_mask

    weighted_accuracy_metric.update_state(y_true, y_pred, sample_weight=energies_times_mask)
    unweighted_accuracy_metric.update_state(y_true, y_pred, sample_weight=valid_mask)

    return unweighted_accuracy_metric.result(), weighted_accuracy_metric.result()


# =======================================================================================================================
# =======================================================================================================================


# ============ CALLBACKS ================================================================================

"""
class CustomLRScheduler(tf.keras.callbacks.Callback):

    def __init__(
        self,
        optim_lr,  # =LR,
        lr_max,  # =0.000015 * train_steps * BATCH_SIZE,
        lr_min,  # =1e-7,
        lr_ramp_ep,  # =3,
        lr_sus_ep,  # =0,
        lr_decay,  # =0.7,
        verbose,
        **kwargs,
    ):
        super(CustomLRScheduler, self).__init__()

        self.optim_lr = optim_lr
        # self.lr_start = lr_start
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_ramp_ep = lr_ramp_ep
        self.lr_sus_ep = lr_sus_ep
        self.lr_decay = lr_decay
        self.verbose = verbose

    def _update_lr(self, epoch):
        if epoch < self.lr_ramp_ep:
            lr = (self.lr_max - self.optim_lr) / self.lr_ramp_ep * epoch + self.optim_lr

        elif epoch < self.lr_ramp_ep + self.lr_sus_ep:
            lr = self.lr_max

        else:
            lr = (self.lr_max - self.lr_min) * self.lr_decay ** (
                epoch - self.lr_ramp_ep - self.lr_sus_ep
            ) + self.lr_min

        return lr

    def on_epoch_begin(self, epoch, logs=None):

        logs = logs or {}
        logs["lr"] = float(self.optim_lr.numpy())

        old_lr = self.optim_lr.numpy()
        new_lr = self._update_lr(epoch)
        self.optim_lr.assign(new_lr)
        if self.verbose > 0:
            print(
                f"\nEpoch {epoch}: Updating learning rate from {old_lr:.4e} to {self.optim_lr.numpy():.4e}"
            )
"""

# =======================================================================================================================
