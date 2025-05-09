import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from ..utils.to_numpy import SENTINEL_NO_DATA, POINT_TYPE_ENCODING, POINT_TYPE_LABELS, event_array_dtype
from numpy.typing import NDArray


def visualize_sample(
    training_sample: NDArray[event_array_dtype], save_path: str = None, normalized: bool = False
):
    if normalized:
        norm = "normalized_"
    else:
        norm = ""
    ax = plt.figure().add_subplot(projection="3d")
    # ax.scatter(0,0,0, label="Origin")
    ax.set_title(f"JZ4, event: {training_sample[0]['eventNumber']}")

    # Focal Track
    focal_hits = training_sample[
        training_sample["category"] == POINT_TYPE_ENCODING["focal_track"]
    ]
    print(f"{len(focal_hits)} focal hits")
    ax.plot(
        focal_hits[f"{norm}x"],
        focal_hits[f"{norm}y"],
        focal_hits[f"{norm}z"],
        label="Focal Track",
    )

    # Non Focal Tracks
    non_focal_hits = training_sample[
        training_sample["category"] == POINT_TYPE_ENCODING["non_focal_track"]
    ]
    unique_tracks = np.unique(non_focal_hits["track_num"])

    # Plot each non-focal track individually
    for i, track_num in enumerate(unique_tracks):
        if track_num != 0:  # focal track
            track_hits = non_focal_hits[non_focal_hits["track_num"] == track_num]
            ax.plot(
                track_hits[f"{norm}x"],
                track_hits[f"{norm}y"],
                track_hits[f"{norm}z"],
                color="r",
                linestyle="--",
            )

    # Cells
    cells = training_sample[training_sample["category"] == POINT_TYPE_ENCODING["cell"]]
    if len(cells) > 0:
        scatter = ax.scatter(
            cells[f"{norm}x"],
            cells[f"{norm}y"],
            cells[f"{norm}z"],
            label="Cells",
            c=np.abs(cells[f"{norm}cell_E"]) + 0.00001,  #
            norm=LogNorm(),
            cmap="viridis",
        )

        plt.colorbar(scatter, ax=ax, label=f"{norm}cell_e (Log Scale)")
    ax.legend()
    if save_path is not None:
        plt.savefig(save_path, dpi=500)
        print(f"Saved to {save_path}")
    return ax

def visualize_inference(
        training_sample: NDArray[event_array_dtype],
        prediction: np.ndarray[float],
        save_path: str = None,
        normalized: bool = False
):
    # TODO: visualize subtraction??
    raise NotImplementedError("Not implemented yet")
    if normalized:
        norm = "normalized_"
    else:
        norm = ""
    ax = plt.figure().add_subplot(projection="3d")
    # ax.scatter(0,0,0, label="Origin")
    ax.set_title(f"JZ4, event: {training_sample[0]['eventNumber']}")

    # Focal Track
    focal_hits = training_sample[
        training_sample["category"] == POINT_TYPE_ENCODING["focal_track"]
        ]
    print(f"{len(focal_hits)} focal hits")
    ax.plot(
        focal_hits[f"{norm}x"],
        focal_hits[f"{norm}y"],
        focal_hits[f"{norm}z"],
        label="Focal Track",
    )

    # Non Focal Tracks
    non_focal_hits = training_sample[
        training_sample["category"] == POINT_TYPE_ENCODING["non_focal_track"]
        ]
    unique_tracks = np.unique(non_focal_hits["track_num"])

    # Plot each non-focal track individually
    for i, track_num in enumerate(unique_tracks):
        if track_num != 0:  # focal track
            track_hits = non_focal_hits[non_focal_hits["track_num"] == track_num]
            ax.plot(
                track_hits[f"{norm}x"],
                track_hits[f"{norm}y"],
                track_hits[f"{norm}z"],
                color="r",
                linestyle="--",
            )

    # Cells
    cells = training_sample[training_sample["category"] == POINT_TYPE_ENCODING["cell"]]
    if len(cells) > 0:
        scatter = ax.scatter(
            cells[f"{norm}x"],
            cells[f"{norm}y"],
            cells[f"{norm}z"],
            label="Cells",
            c=np.abs(cells[f"{norm}cell_E"]) + 0.00001,  #
            norm=LogNorm(),
            cmap="viridis",
        )

        plt.colorbar(scatter, ax=ax, label=f"{norm}cell_e (Log Scale)")
    ax.legend()
    if save_path is not None:
        plt.savefig(save_path, dpi=500)
        print(f"Saved to {save_path}")
    return ax
