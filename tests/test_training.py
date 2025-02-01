from prod.JetPointNet import (
    masked_weighted_loss,
    masked_weighted_accuracy_multi_target,
    masked_weighted_accuracy_single_target,
)
import tensorflow as tf
import pytest
from prod.utils.to_numpy import POINT_TYPE_ENCODING

possible_transforms = [None, "none", "absolute", "square", "normalize", "threshold"]
n_targets_to_test = [1, 2, 3, 4]


def setup_test_data(
    batch_size: int, n_points: int, n_targets: int
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    y_true = tf.random.uniform(
        shape=(batch_size, n_points, n_targets), minval=0, maxval=1, dtype=tf.float32
    )
    y_pred = tf.random.uniform(
        shape=(batch_size, n_points, n_targets), minval=0, maxval=1, dtype=tf.float32
    )
    energies = tf.random.uniform(
        shape=(batch_size, n_points), minval=0, maxval=1, dtype=tf.float32
    )
    x_class = tf.constant(
        [[POINT_TYPE_ENCODING["cell"]] * n_points] * batch_size, dtype=tf.int32
    )
    return y_true, y_pred, energies, x_class


@pytest.mark.parametrize("transform", possible_transforms)
@pytest.mark.parametrize("n_targets", n_targets_to_test)
def test_masked_weighted_loss_dimensions(
    transform: str, n_targets: int, batch_size=10, n_points=100
):
    y_true, y_pred, energies, x_class = setup_test_data(batch_size, n_points, n_targets)

    if n_targets == 1:
        loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    else:
        loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    loss = masked_weighted_loss(
        y_true=y_true,
        y_pred=y_pred,
        energies=energies,
        loss_function=loss_function,
        x_class=x_class,
        transform=transform,
        energy_threshold=0.5,
    )

    assert isinstance(loss, tf.Tensor), "Loss should be a tf.Tensor"
    assert loss.shape == (), f"Expected scalar shape from loss. Got shape={loss.shape}"


def test_masked_weighted_loss_raises_for_invalid_transform():
    batch_size = 4
    y_true, y_pred, energies, x_class = setup_test_data(batch_size, 1, 1)
    loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    with pytest.raises(ValueError) as exc_info:
        masked_weighted_loss(
            y_true=y_true,
            y_pred=y_pred,
            energies=energies,
            loss_function=loss_function,
            x_class=x_class,
            transform="invalid_transform",
        )
    assert "Unknown transform value" in str(exc_info.value)


@pytest.mark.parametrize("transform", possible_transforms)
@pytest.mark.parametrize("n_targets", n_targets_to_test)
def test_masked_weighted_accuracy_dimensions(transform, n_targets, batch_size=10):
    y_true, y_pred, energies, x_class = setup_test_data(batch_size, 1, n_targets)

    unweighted_accuracy_metric = tf.keras.metrics.BinaryAccuracy(threshold=0.0)
    weighted_accuracy_metric = tf.keras.metrics.BinaryAccuracy(threshold=0.0)

    if n_targets == 1:
        masked_weighted_accuracy = masked_weighted_accuracy_single_target
    else:
        masked_weighted_accuracy = masked_weighted_accuracy_multi_target

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
    batch_size = 8
    n_targets = 1
    y_true, y_pred, energies, x_class = setup_test_data(batch_size, 1, n_targets)

    unweighted_accuracy_metric = tf.keras.metrics.BinaryAccuracy()
    weighted_accuracy_metric = tf.keras.metrics.BinaryAccuracy()

    if n_targets == 1:
        masked_weighted_accuracy = masked_weighted_accuracy_single_target
    else:
        masked_weighted_accuracy = masked_weighted_accuracy_multi_target

    with pytest.raises(ValueError) as exc_info:
        masked_weighted_accuracy(
            y_true=y_true,
            y_pred=y_pred,
            energies=energies,
            x_class=x_class,
            unweighted_accuracy_metric=unweighted_accuracy_metric,
            weighted_accuracy_metric=weighted_accuracy_metric,
            transform="invalid_transform",
        )
    assert "Unknown transform value" in str(exc_info.value)


@pytest.mark.parametrize("transform", possible_transforms)
@pytest.mark.parametrize("n_targets", n_targets_to_test)
def test_compare_performance_and_check_loss(n_targets: int, transform: str):
    batch_size = 10
    n_points = 100
    y_true, y_pred, energies, x_class = setup_test_data(batch_size, n_points, n_targets)

    if n_targets == 1:
        loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    else:
        loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    loss = masked_weighted_loss(
        y_true=y_true,
        y_pred=y_pred,
        energies=energies,
        loss_function=loss_function,
        x_class=x_class,
        transform=transform,
        energy_threshold=0.5,
    )

    assert isinstance(loss, tf.Tensor), "Loss should be a tf.Tensor"
    assert loss.shape == (), f"Expected scalar shape from loss. Got shape={loss.shape}"
    assert loss.numpy() > 0, "Loss should be greater than 0"


@pytest.mark.parametrize("transform", possible_transforms)
@pytest.mark.parametrize("n_targets", n_targets_to_test)
def test_masked_weighted_accuracy_perfect_prediction(n_targets: int, transform: str):
    batch_size = 5
    n_points = 10
    y_true, y_pred, energies, x_class = setup_test_data(batch_size, n_points, n_targets)
    y_pred = y_true  # Set y_pred to be the same as y_true for perfect prediction

    unweighted_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
    weighted_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

    if n_targets == 1:
        masked_weighted_accuracy = masked_weighted_accuracy_single_target
    else:
        masked_weighted_accuracy = masked_weighted_accuracy_multi_target

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

    assert (
        unweighted_acc.numpy() == 1.0
    ), "Unweighted accuracy should be 1.0 for perfect prediction"
    assert (
        weighted_acc.numpy() == 1.0
    ), "Weighted accuracy should be 1.0 for perfect prediction"


@pytest.mark.parametrize("transform", possible_transforms)
@pytest.mark.parametrize("n_targets", n_targets_to_test)
def test_loss_decreases_with_better_predictions(n_targets: int, transform: str):
    batch_size = 5
    n_points = 10

    # Generate test data
    y_true, _, energies, x_class = setup_test_data(batch_size, n_points, n_targets)

    y_pred_close = y_true + tf.constant(0.1, shape=y_true.shape)
    y_pred_far = tf.constant(1, shape=y_true.shape)

    if n_targets == 1:
        loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    else:
        loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # Calculate the loss for both sets of predictions
    loss_close = masked_weighted_loss(
        y_true=y_true,
        y_pred=y_pred_close,
        energies=energies,
        loss_function=loss_function,
        x_class=x_class,
        transform=transform,
        energy_threshold=0.5,
    )

    loss_far = masked_weighted_loss(
        y_true=y_true,
        y_pred=y_pred_far,
        energies=energies,
        loss_function=loss_function,
        x_class=x_class,
        transform=transform,
        energy_threshold=0.5,
    )

    # Assert that the loss is lower for the closer predictions
    assert (
        loss_close.numpy() < loss_far.numpy()
    ), "Loss should be lower for better predictions"


@pytest.mark.parametrize("transform", possible_transforms)
@pytest.mark.parametrize("n_targets", [1, 2, 3, 4])
def test_accuracy_increases_with_better_predictions(n_targets: int, transform: str):
    batch_size = 5
    n_points = 10

    # Generate test data
    y_true, _, energies, x_class = setup_test_data(batch_size, n_points, n_targets)

    y_pred_close = y_true + tf.constant(0.1, shape=y_true.shape)
    y_pred_far = tf.constant(1, shape=y_true.shape)

    unweighted_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
    weighted_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

    if n_targets == 1:
        masked_weighted_accuracy = masked_weighted_accuracy_single_target
    else:
        masked_weighted_accuracy = masked_weighted_accuracy_multi_target

    # Calculate the accuracy for both sets of predictions
    unweighted_acc_close, weighted_acc_close = masked_weighted_accuracy(
        y_true=y_true,
        y_pred=y_pred_close,
        energies=energies,
        x_class=x_class,
        unweighted_accuracy_metric=unweighted_accuracy_metric,
        weighted_accuracy_metric=weighted_accuracy_metric,
        transform=transform,
        energy_threshold=0.5,
    )

    unweighted_acc_far, weighted_acc_far = masked_weighted_accuracy(
        y_true=y_true,
        y_pred=y_pred_far,
        energies=energies,
        x_class=x_class,
        unweighted_accuracy_metric=unweighted_accuracy_metric,
        weighted_accuracy_metric=weighted_accuracy_metric,
        transform=transform,
        energy_threshold=0.5,
    )

    # Assert that the accuracy is higher for the closer predictions
    assert (
        unweighted_acc_close.numpy() > unweighted_acc_far.numpy()
    ), "Unweighted accuracy should be higher for better predictions"
    assert (
        weighted_acc_close.numpy() > weighted_acc_far.numpy()
    ), "Weighted accuracy should be higher for better predictions"


# @pytest.mark.parametrize("transform", possible_transforms)
# def test_accuracy_increases_with_better_predictions_single_target(transform: str):
#     batch_size = 5
#     n_points = 10
#     n_targets = 1
#
#     # Generate test data
#     y_true, _, energies, x_class = setup_test_data(batch_size, n_points, n_targets)
#
#     y_pred_close = y_true + tf.constant(0.1, shape=y_true.shape)
#     y_pred_far = tf.constant(1, shape=y_true.shape)
#
#     if n_targets == 1:
#         unweighted_accuracy_metric = tf.keras.metrics.BinaryAccuracy()
#         weighted_accuracy_metric = tf.keras.metrics.BinaryAccuracy()
#     else:
#         unweighted_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
#         weighted_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
#
#     # Calculate the accuracy for both sets of predictions
#     unweighted_acc_close, weighted_acc_close = masked_weighted_accuracy(
#         y_true=y_true,
#         y_pred=y_pred_close,
#         energies=energies,
#         x_class=x_class,
#         unweighted_accuracy_metric=unweighted_accuracy_metric,
#         weighted_accuracy_metric=weighted_accuracy_metric,
#         transform=transform,
#         energy_threshold=0.5,
#     )
#
#     unweighted_acc_far, weighted_acc_far = masked_weighted_accuracy_single_target(
#         y_true=y_true,
#         y_pred=y_pred_far,
#         energies=energies,
#         x_class=x_class,
#         unweighted_accuracy_metric=unweighted_accuracy_metric,
#         weighted_accuracy_metric=weighted_accuracy_metric,
#         transform=transform,
#         energy_threshold=0.5,
#     )
#
#     # Assert that the accuracy is higher for the closer predictions
#     assert (
#         unweighted_acc_close.numpy() > unweighted_acc_far.numpy()
#     ), "Unweighted accuracy should be higher for better predictions"
#     assert (
#         weighted_acc_close.numpy() > weighted_acc_far.numpy()
#     ), "Weighted accuracy should be higher for better predictions"

if __name__ == "__main__":
    test_accuracy_increases_with_better_predictions(1, "none")
