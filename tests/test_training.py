from prod.JetPointNet import masked_weighted_loss, masked_weighted_accuracy
import tensorflow as tf
import pytest
from prod.utils.to_numpy import POINT_TYPE_ENCODING


@pytest.mark.parametrize(
    "transform",
    [None, "none", "absolute", "square", "normalize", "standardize", "threshold"],
)
@pytest.mark.parametrize("n_targets", [1, 2, 3, 4])
def test_masked_weighted_loss_dimensions(transform: str, n_targets: int, batch_size=10):
    """
    Test that masked_weighted_loss returns a scalar loss
    and handles inputs of various dimensions correctly.
    """

    # Generate dummy data
    batch_size = 10
    # y_true and y_pred typically have shape (batch_size, targets) or (batch_size, targets)
    y_true = tf.random.uniform(
        shape=(batch_size, n_targets), minval=0, maxval=1, dtype=tf.float32
    )
    y_pred = tf.random.normal(shape=(batch_size, n_targets))  # model outputs

    # energies and x_class with shape = (batch_size,), one value per sample
    energies = tf.random.uniform(shape=(batch_size,), minval=-5.0, maxval=5.0)
    x_class = tf.constant(
        [POINT_TYPE_ENCODING["cell"]] * batch_size, dtype=tf.int32
    )  # all 'cell'

    if n_targets == 1:
        loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    else:
        loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # Because "threshold" requires an energy_threshold, we just pass a value (e.g., 0.0).
    loss = masked_weighted_loss(
        y_true=y_true,
        y_pred=y_pred,
        energies=energies,
        loss_function=loss_function,
        x_class=x_class,
        transform=transform,
        energy_threshold=0.5,
    )

    # Check that the output is a scalar
    # Typically, tf.keras loss functions return scalar shape []
    assert isinstance(loss, tf.Tensor), "Loss should be a tf.Tensor"
    assert loss.shape == (), f"Expected scalar shape from loss. Got shape={loss.shape}"


def test_masked_weighted_loss_raises_for_invalid_transform():
    """
    Test that masked_weighted_loss raises a ValueError when an invalid transform is passed.
    """
    batch_size = 4
    y_true = tf.random.uniform(shape=(batch_size,), minval=0, maxval=2, dtype=tf.int32)
    y_pred = tf.random.normal(shape=(batch_size, 1))
    energies = tf.random.uniform(shape=(batch_size,), minval=-5.0, maxval=5.0)
    x_class = tf.constant([1] * batch_size, dtype=tf.int32)
    loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    with pytest.raises(ValueError) as exc_info:
        masked_weighted_loss(
            y_true=y_true,
            y_pred=y_pred,
            energies=energies,
            loss_function=loss_function,
            x_class=x_class,
            transform="invalid_transform",  # invalid
        )
    assert "Unknown transform value" in str(exc_info.value)


@pytest.mark.parametrize(
    "transform",
    [None, "none", "absolute", "square", "normalize", "standardize", "threshold"],
)
@pytest.mark.parametrize("n_targets", [1, 2, 3, 4])
def test_masked_weighted_accuracy_dimensions(transform, n_targets, batch_size=10):
    """
    Test that masked_weighted_accuracy returns two scalar metrics
    (unweighted_accuracy, weighted_accuracy).
    """

    # y_true and y_pred typically have shape (batch_size, targets) or (batch_size, targets)
    y_true = tf.random.uniform(
        shape=(batch_size, n_targets), minval=0, maxval=1, dtype=tf.float32
    )
    y_pred = tf.random.normal(shape=(batch_size, n_targets))  # model outputs

    # energies and x_class with shape = (batch_size,), one value per sample
    energies = tf.random.uniform(shape=(batch_size,), minval=-5.0, maxval=5.0)
    x_class = tf.constant(
        [POINT_TYPE_ENCODING["cell"]] * batch_size, dtype=tf.int32
    )  # all 'cell'

    # Define a simple loss function
    unweighted_accuracy_metric = tf.keras.metrics.BinaryAccuracy(threshold=0.0)
    weighted_accuracy_metric = tf.keras.metrics.BinaryAccuracy(threshold=0.0)

    unweighted_acc, weighted_acc = masked_weighted_accuracy(
        y_true=y_true,
        y_pred=y_pred,
        energies=energies,
        x_class=x_class,
        unweighted_accuracy_metric=unweighted_accuracy_metric,
        weighted_accuracy_metric=weighted_accuracy_metric,
        transform=transform,
        energy_threshold=0.5,
    )

    # Both returned values should be scalar Tensors
    assert isinstance(
        unweighted_acc, tf.Tensor
    ), "Unweighted accuracy should be a tf.Tensor"
    assert (
        unweighted_acc.shape == ()
    ), f"Unweighted accuracy should be a scalar. Got shape={unweighted_acc.shape}"

    assert isinstance(
        weighted_acc, tf.Tensor
    ), "Weighted accuracy should be a tf.Tensor"
    assert (
        weighted_acc.shape == ()
    ), f"Weighted accuracy should be a scalar. Got shape={weighted_acc.shape}"


def test_masked_weighted_accuracy_raises_for_invalid_transform():
    """
    Test that masked_weighted_accuracy raises a ValueError when an invalid transform is passed.
    """
    batch_size = 8
    y_true = tf.random.uniform(shape=(batch_size,), minval=0, maxval=2, dtype=tf.int32)
    y_pred = tf.random.normal(shape=(batch_size,))
    energies = tf.random.uniform(shape=(batch_size,), minval=-5.0, maxval=5.0)
    x_class = tf.constant([1] * batch_size, dtype=tf.int32)

    unweighted_accuracy_metric = tf.keras.metrics.BinaryAccuracy()
    weighted_accuracy_metric = tf.keras.metrics.BinaryAccuracy()

    with pytest.raises(ValueError) as exc_info:
        masked_weighted_accuracy(
            y_true=y_true,
            y_pred=y_pred,
            energies=energies,
            x_class=x_class,
            unweighted_accuracy_metric=unweighted_accuracy_metric,
            weighted_accuracy_metric=weighted_accuracy_metric,
            transform="invalid_transform",  # invalid
        )
    assert "Unknown transform value" in str(exc_info.value)
