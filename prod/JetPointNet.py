"""
Adapted From
https://github.com/lattice-ai/pointnet/tree/master

Original Architecture From Pointnet Paper:
https://arxiv.org/pdf/1612.00593.pdf
"""

import os
import random
import sys
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

try:
    import tensorflow.keras as keras
except ImportError:
    import keras

REPO_PATH = Path.home() / "workspace/jetpointnet"
SCRIPT_PATH = REPO_PATH / "python_scripts"
sys.path.append(str(SCRIPT_PATH))

SENTINEL_NO_DATA = -1
POINT_TYPE_LABELS = {
    0: "focal_track",
    1: "cell",
    2: "non_focal_track",
    SENTINEL_NO_DATA: "padding",
}
POINT_TYPE_ENCODING = {v: k for k, v in POINT_TYPE_LABELS.items()}


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
        self.model.save(f"JetPointNet_{epoch}.hd5".format(epoch))


class CustomMaskingLayer(keras.layers.Layer):
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


class OrthogonalRegularizer(keras.regularizers.OrthogonalRegularizer):
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
    return keras.backend.cast(x > 0, dtype=tf.float32)


# =======================================================================================================================
# =======================================================================================================================


# =======================================================================================================================
# ============ Main Model Blocks ========================================================================================


def conv_mlp(
    input_tensor, filters, dropout_rate=None, apply_attention=False, name=None
):
    if name is not None:
        x = keras.layers.Conv1D(
            filters=filters, kernel_size=1, activation="relu", name=name
        )(input_tensor)
    else:
        x = keras.layers.Conv1D(filters=filters, kernel_size=1, activation="relu")(
            input_tensor
        )
    x = keras.layers.BatchNormalization()(x)

    if apply_attention:
        # Self-attention
        attention_output_self = keras.layers.MultiHeadAttention(
            num_heads=2, key_dim=filters
        )(x, x)
        attention_output_self = keras.layers.LayerNormalization()(
            attention_output_self + x
        )

        # Cross-attention
        attention_output_cross = keras.layers.MultiHeadAttention(
            num_heads=2, key_dim=filters
        )(attention_output_self, x)
        attention_output_cross = keras.layers.LayerNormalization()(
            attention_output_cross + attention_output_self
        )

        x = attention_output_cross

    if dropout_rate is not None:
        x = keras.layers.Dropout(dropout_rate)(x)

    return x


def dense_block(input_tensor, units, dropout_rate=None, regularizer=None):
    x = keras.layers.Dense(units, kernel_regularizer=regularizer)(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    if dropout_rate is not None:
        x = keras.layers.Dropout(dropout_rate)(x)
    return x


def TNet(
    input_tensor, size, add_regularization=False
):  # JH: why don't we add regularization? Not an issue I don't think, but super strange
    # size is either 6 for the first TNet or 64 for the second
    x = conv_mlp(input_tensor, 64)
    x = conv_mlp(x, 128)
    x = conv_mlp(x, 1024)
    x = keras.layers.GlobalMaxPooling1D()(x)
    x = dense_block(x, 512)
    x = dense_block(x, 256)
    if add_regularization:
        reg = OrthogonalRegularizer(size)
    else:
        reg = None
    x = dense_block(x, size * size, regularizer=reg)
    x = keras.layers.Reshape((size, size))(x)
    return x


def PointNetSegmentation(
    num_points: int,
    num_features: int,
    num_classes: int,
    output_activation_function: str,
    model_version: int,
) -> keras.Model:
    """
    PointNet model for segmentation of point clouds.

    Args:
        num_points: The number of points in each batch.
        num_features: The number of features for each point.
        num_classes: The number of classes to predict.
        output_activation_function:
        model_version:

    Returns:
        keras.Model: The model.
    """
    input_points = keras.Input(shape=(num_points, num_features))

    # Masking layer to ignore points with the last feature index as -1
    masks = keras.layers.Lambda(
        lambda x: tf.not_equal(x[:, :, -1], 1), output_shape=(num_points,)
    )(input_points)
    masks = keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(
        masks
    )  # Cast boolean to float for multiplication
    masks = keras.layers.Lambda(lambda x: tf.expand_dims(x, -1))(
        masks
    )  # Expand dimensions to apply mask

    # Apply mask
    input_points_masked = keras.layers.Multiply()([input_points, masks])

    # energy = keras.layers.Lambda(lambda x: tf.expand_dims(x[:, :, 4], -1), name='e')(input_points_masked)

    # T-Net for input transformation
    input_tnet = TNet(
        input_points_masked, num_features
    )  # Assuming TNet is properly defined elsewhere
    x = keras.layers.Dot(axes=(2, 1))([input_points_masked, input_tnet])
    x = conv_mlp(x, 96)  # JH: This should be 64
    x = conv_mlp(x, 96)  # JH: This should be 64
    point_features = x

    # T-Net for feature transformation
    feature_tnet = TNet(x, 96, add_regularization=True)
    x = keras.layers.Dot(axes=(2, 1))([x, feature_tnet])
    x = conv_mlp(x, 128)  # JH: this should be 64?
    x = conv_mlp(x, 256)  # JH: this should be 128?
    x = conv_mlp(x, 1024)

    # Get global features and expand
    global_feature = keras.layers.GlobalMaxPooling1D(name="GlobalPooling")(x)
    global_feature_expanded = keras.layers.Lambda(lambda x: tf.expand_dims(x, 1))(
        global_feature
    )
    global_feature_expanded = keras.layers.Lambda(
        lambda x: tf.tile(x, [1, num_points, 1])
    )(global_feature_expanded)

    # Segmentation head
    if model_version == 0:  # ~5M params
        c = keras.layers.Concatenate()([point_features, global_feature_expanded])

        c = conv_mlp(c, 512, apply_attention=False)
        c = conv_mlp(c, 256, apply_attention=False)

        c = conv_mlp(c, 128, dropout_rate=0.3)
    elif model_version == 1:  # ~6M params
        c = keras.layers.Concatenate()([point_features, global_feature_expanded])

        c = conv_mlp(c, 1024, apply_attention=False)

        c = conv_mlp(c, 512, apply_attention=False)
        c = conv_mlp(c, 256, apply_attention=False)
        c = conv_mlp(c, 128, apply_attention=False)

        c = conv_mlp(c, 128, dropout_rate=0.3)
    elif model_version == 2:  # ~7M params
        c = keras.layers.Concatenate()([point_features, global_feature_expanded])

        c = conv_mlp(c, 1024, apply_attention=False)

        c = conv_mlp(c, 1024, apply_attention=False)
        c = conv_mlp(c, 512, apply_attention=False)
        c = conv_mlp(c, 256, apply_attention=False)

        c = conv_mlp(c, 256, dropout_rate=0.3)
    elif model_version == 3:
        c = keras.layers.Concatenate()([point_features, global_feature_expanded])

        c = conv_mlp(c, 2048, apply_attention=False)
        c = conv_mlp(c, 1024, apply_attention=False)
        c = conv_mlp(c, 1024, apply_attention=False)
        c = conv_mlp(c, 512, apply_attention=False)
        c = conv_mlp(c, 512, apply_attention=False)

        c = conv_mlp(c, 256, dropout_rate=0.3)
    else:
        raise Exception("INVALID MODEL VERSION")

    segmentation_output = keras.layers.Conv1D(
        num_classes, kernel_size=1, activation=output_activation_function, name="SEG"
    )(c)

    model = keras.Model(inputs=input_points, outputs=segmentation_output)

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
    loss_function: keras.losses.Loss,
    x_class: tf.Tensor,
    transform: None | str = None,
    energy_threshold: float = 0,
) -> tf.Tensor:
    """
    Computes the masked weighted loss of predictions.

    Parameters:
    y_true (tf.Tensor): True labels. Shape: (batch_size, num_points, num_classes)
    y_pred (tf.Tensor): Predicted labels. Shape: (batch_size, num_points, num_classes)
    energies (tf.Tensor): Weights for each prediction. Shape: (batch_size, num_points)
    loss_function: (keras.losses.Loss): The loss function to call with the model outputs.
    x_class: (tf.Tensor): The point-type for each cell, CELL, PAD, TRACK, etc. Shape: (batch_size, num_points)
    transform (str, optional): Transformation to apply to energies. Possible values:
        - None: no transformation (default).
        - "absolute": absolute value.
        - "square": square.
        - "normalize": batch-normalize to zero mean and unit variance.
        - "standardize": batch-standardize to zero mean and unit variance.
        - "threshold": threshold at 0 --> discard contributions by negative energies.
    energy_threshold (float, optional): the threshold to cut off energy weighting if "threshold" is the transform

    Returns:
    tf.Tensor: standardized loss. Single value of shape ().
    """
    # shape of y_true is (batch_size, num_points, num_classes)
    assert len(y_true.shape) == 3
    assert len(y_pred.shape) == 3
    assert len(energies.shape) == 2
    assert len(x_class.shape) == 2
    assert y_true.shape == y_pred.shape
    assert x_class.shape[1] == y_true.shape[1]
    assert y_true.shape[1] == energies.shape[1]

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
        # standardize does not work well with negative energies
        # case "standardize":
        #     energies = (energies - tf.reduce_mean(energies)) / (
        #         tf.math.reduce_std(energies) + 1e-5
        #     )
        case "threshold":
            energies = tf.cast(tf.greater(energies, energy_threshold), tf.float32)
        case None | "none":
            pass
        case _:
            raise ValueError(f"Unknown transform value: {transform}")

    valid_mask = tf.equal(x_class, POINT_TYPE_ENCODING["cell"])
    valid_mask = tf.cast(valid_mask, tf.float32)

    energies_times_mask = energies * valid_mask

    weighted_loss = loss_function(y_true, y_pred, sample_weight=energies_times_mask)

    return weighted_loss


@tf.autograph.experimental.do_not_convert
def masked_weighted_accuracy(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    energies: tf.Tensor,
    x_class: tf.Tensor,
    unweighted_accuracy_metric: keras.metrics.Metric,
    weighted_accuracy_metric: keras.metrics.Metric,
    transform: None | str = None,
    energy_threshold: float = 0,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Computes the masked weighted and unweighted accuracy of predictions.

    Parameters:
    y_true (tf.Tensor): True labels. Shape: (batch_size, num_points, num_classes)
    y_pred (tf.Tensor): Predicted labels. Shape: (batch_size, num_points, num_classes)
    energies (tf.Tensor): Weights for each prediction. Shape: (batch_size, num_points)
    x_class: (tf.Tensor): The point-type for each cell, CELL, PAD, TRACK, etc. Shape: (batch_size, num_points)
    unweighted_accuracy_metric (keras.metrics.Metric): The metric to be called with outputs.
    weighted_accuracy_metric (keras.metrics.Metric): The metric to be called with outputs
    transform (str, optional): Transformation to apply to energies. Possible values:
        - None: no transformation (default).
        - "absolute": absolute value.
        - "square": square.
        - "normalize": batch-normalize to zero mean and unit variance.
        - "standardize": batch-standardize to zero mean and unit variance.
        - "threshold": threshold at 0 --> discard contributions by negative energies.
    energy_threshold (float, optional): the threshold to cutoff energy weighting if "threshold" is the transform.

    Returns:
        tuple[tf.Tensor, tf.Tensor]: unweighted accuracy, weighted accuracy, both of shape ().
    """
    if y_true.shape[-1] == 1:
        return masked_weighted_accuracy_single_target(
            y_true,
            y_pred,
            energies,
            x_class,
            unweighted_accuracy_metric,
            weighted_accuracy_metric,
            transform,
            energy_threshold,
        )
    else:
        return masked_weighted_accuracy_multi_target(
            y_true,
            y_pred,
            energies,
            x_class,
            unweighted_accuracy_metric,
            weighted_accuracy_metric,
            transform,
            energy_threshold,
        )


@tf.autograph.experimental.do_not_convert
def masked_weighted_accuracy_multi_target(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    energies: tf.Tensor,
    x_class: tf.Tensor,
    unweighted_accuracy_metric: keras.metrics.Metric,
    weighted_accuracy_metric: keras.metrics.Metric,
    transform: None | str = None,
    energy_threshold: float = 0,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Computes the masked weighted and unweighted accuracy of predictions.

    Parameters:
    y_true (tf.Tensor): True labels. Shape: (batch_size, num_points, num_classes)
    y_pred (tf.Tensor): Predicted labels. Shape: (batch_size, num_points, num_classes)
    energies (tf.Tensor): Weights for each prediction. Shape: (batch_size, num_points)
    x_class: (tf.Tensor): The point-type for each cell, CELL, PAD, TRACK, etc. Shape: (batch_size, num_points)
    unweighted_accuracy_metric (keras.metrics.Metric): The metric to be called with outputs.
    weighted_accuracy_metric (keras.metrics.Metric): The metric to be called with outputs
    transform (str, optional): Transformation to apply to energies. Possible values:
        - None: no transformation (default).
        - "absolute": absolute value.
        - "square": square.
        - "normalize": batch-normalize to zero mean and unit variance.
        - "standardize": batch-standardize to zero mean and unit variance.
        - "threshold": threshold at 0 --> discard contributions by negative energies.
    energy_threshold (float, optional): the threshold to cutoff energy weighting if "threshold" is the transform.

    Returns:
        tuple[tf.Tensor, tf.Tensor]: unweighted accuracy, weighted accuracy, both of shape ().
    """
    # assert last dim is singlet
    assert y_true.shape[-1] != 1

    assert len(y_true.shape) == 3
    assert len(y_pred.shape) == 3
    assert len(energies.shape) == 2
    assert len(x_class.shape) == 2
    assert y_true.shape == y_pred.shape
    assert x_class.shape[1] == y_true.shape[1]

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
        # standardize does not work well with negative energies
        # case "standardize":
        #     energies = (energies - tf.reduce_mean(energies)) / (
        #         tf.math.reduce_std(energies) + 1e-5
        #     )
        case "threshold":
            energies = tf.cast(tf.greater(energies, energy_threshold), tf.float32)
        case None | "none":
            pass
        case _:
            raise ValueError(f"Unknown transform value: {transform}")

    valid_mask = tf.equal(x_class, POINT_TYPE_ENCODING["cell"])
    valid_mask = tf.cast(valid_mask, tf.float32)

    energies_times_mask = energies * valid_mask

    weighted_accuracy_metric.update_state(
        y_true, y_pred, sample_weight=energies_times_mask
    )
    unweighted_accuracy_metric.update_state(y_true, y_pred, sample_weight=valid_mask)

    return unweighted_accuracy_metric.result(), weighted_accuracy_metric.result()


@tf.autograph.experimental.do_not_convert
def masked_weighted_accuracy_single_target(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    energies: tf.Tensor,
    x_class: tf.Tensor,
    unweighted_accuracy_metric: keras.metrics.Metric,
    weighted_accuracy_metric: keras.metrics.Metric,
    transform: None | str = None,
    energy_threshold: float = 0,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Computes the masked weighted and unweighted accuracy of predictions.

    Parameters:
    y_true (tf.Tensor): True labels. Shape: (batch_size, num_points, num_classes)
    y_pred (tf.Tensor): Predicted labels. Shape: (batch_size, num_points, num_classes)
    energies (tf.Tensor): Weights for each prediction. Shape: (batch_size, num_points)
    x_class: (tf.Tensor): The point-type for each cell, CELL, PAD, TRACK, etc. Shape: (batch_size, num_points)
    unweighted_accuracy_metric (tf.keras.metrics.Metric): The metric to be called with outputs.
    weighted_accuracy_metric (tf.keras.metrics.Metric): The metric to be called with outputs
    transform (str, optional): Transformation to apply to energies. Possible values:
        - None: no transformation (default).
        - "absolute": absolute value.
        - "square": square.
        - "normalize": batch-normalize to zero mean and unit variance.
        - "standardize": batch-standardize to zero mean and unit variance.
        - "threshold": threshold at 0 --> discard contributions by negative energies.
    energy_threshold (float, optional): the threshold to cutoff energy weighting if "threshold" is the transform.

    Returns:
        tuple[tf.Tensor, tf.Tensor]: unweighted accuracy, weighted accuracy, both of shape ().
    """
    # assert last dim is singlet
    assert y_true.shape[-1] == 1
    assert len(y_true.shape) == 3
    assert len(y_pred.shape) == 3
    assert len(energies.shape) == 2
    assert len(x_class.shape) == 2
    assert y_true.shape == y_pred.shape
    assert x_class.shape[1] == y_true.shape[1]

    y_true_squeezed = tf.squeeze(y_true, axis=-1)
    y_pred_squeezed = tf.squeeze(y_pred, axis=-1)

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
        case "threshold":
            energies = tf.cast(tf.greater(energies, energy_threshold), tf.float32)
        case None | "none":
            pass
        case _:
            raise ValueError(f"Unknown transform value: {transform}")

    valid_mask = tf.equal(x_class, POINT_TYPE_ENCODING["cell"])
    valid_mask = tf.cast(valid_mask, tf.float32)
    energies_times_mask = energies * valid_mask

    # TODO: This is now a per-event accuracy, not per-point, I think this is highly problematic, but I'm not sure how to fix it
    weighted_accuracy_metric.update_state(
        y_true_squeezed,
        y_pred_squeezed,
        sample_weight=tf.reduce_sum(energies_times_mask),
    )
    unweighted_accuracy_metric.update_state(
        y_true_squeezed, y_pred_squeezed, sample_weight=tf.reduce_sum(valid_mask)
    )

    return unweighted_accuracy_metric.result(), weighted_accuracy_metric.result()


# =======================================================================================================================
# =======================================================================================================================


# ============ CALLBACKS ================================================================================

"""
class CustomLRScheduler(keras.callbacks.Callback):

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
