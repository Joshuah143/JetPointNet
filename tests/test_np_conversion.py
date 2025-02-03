import numpy as np
import pytest
import awkward as ak
from pathlib import Path
from numpy.typing import NDArray

from prod.utils.to_numpy import (
    event_to_trainable,
    event_array_dtype,
    POINT_TYPE_ENCODING,
)
from prod.load_from_root_file import load_from_root


@pytest.fixture(scope="module")
def loaded_event():
    if Path(__file__).parent.name == "tests":
        test_files_location = Path(__file__).parent / Path("test_input")
    elif Path(__file__).parent.name == "prod":
        test_files_location = Path(__file__).parent / Path("tests/test_input")
    else:
        raise FileNotFoundError("Cannot find test files")

    test_input_path = test_files_location / "rho.root"
    geo_file = test_files_location / "geo_file.root"
    events_ak = load_from_root(
        root_files_location=test_input_path,
        geo_file=geo_file,
        debug=True,
    )
    focal_ak = events_ak[1]  # 0th event has no tracks
    event_len = 100
    focal_np: NDArray[event_array_dtype] = event_to_trainable(
        event=focal_ak,
        focal_index=0,
        delta_r_max=0.2,
        max_event_len=event_len,
    )
    non_padding = focal_np[focal_np["category"] != POINT_TYPE_ENCODING["padding"]]

    return focal_np, non_padding, event_len


def test_np_shape_and_type(loaded_event):
    focal_np, _, event_len = loaded_event
    assert focal_np.shape == (event_len,)
    assert focal_np.dtype == event_array_dtype


def test_assert_normalizations_within_bounds(loaded_event):
    _, non_padding, _ = loaded_event
    normalized_cols = [
        ("normalized_x", np.float32),
        ("normalized_y", np.float32),
        ("normalized_z", np.float32),
        ("normalized_x_isolated", np.float32),
        ("normalized_y_isolated", np.float32),
        ("normalized_z_isolated", np.float32),
        ("normalized_cell_E", np.float32),
        ("normalized_track_pt", np.float32),
    ]

    for col, dtype in normalized_cols:
        assert np.all(non_padding[col] >= -1), f"{col} has values below -1"
        assert np.all(non_padding[col] <= 1), f"{col} has values above 1"
        assert non_padding[col].dtype == dtype, f"{col} has incorrect dtype"


def test_no_columns_are_empty(loaded_event):
    _, non_padding, _ = loaded_event
    for col in non_padding.dtype.names:
        non_zero_values = non_padding[col][
            non_padding[col] != POINT_TYPE_ENCODING["padding"]
        ]
        assert len(non_zero_values) > 0, f"Column {col} is empty"
        assert np.any(non_padding[col] != 0), f"Column {col} only contains zeros"
