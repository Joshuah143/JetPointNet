import awkward as ak
import numpy as np
from loguru import logger as log
from .coordinate_conversions import calculate_delta_r

SENTINEL_NO_DATA = -1
POINT_TYPE_LABELS = {
    0: "focal_track",
    1: "cell",
    2: "non_focal_track",
    SENTINEL_NO_DATA: "padding",
}
POINT_TYPE_ENCODING = {v: k for k, v in POINT_TYPE_LABELS.items()}

# TODO: add tests for this file

event_array_dtype = np.dtype(
    [
        ("runNumber", np.int32),
        ("eventNumber", np.int32),
        ("cell_ID", np.int32),
        ("track_index", np.int32),
        # SENTINEL_NO_DATA if non-truth event
        ("truth_cell_focal_fraction_energy", np.float32),
        ("truth_cell_focal_energy", np.float32),
        ("truth_cell_focal_observed_fraction_energy", np.float32),
        # ('truth_cell_non_focal_fraction_energy', np.float32), TODO: add this into the training data for multi-label classification
        # ('truth_cell_neutral_fraction_energy', np.float32), TODO: add this into the training data for multi-label classification
        ("truth_cell_total_energy", np.float32),
        ("category", np.int8),
        ("track_num", np.int32),
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("delta_r", np.float32),
        ("eta", np.float32),
        ("phi", np.float32),
        # ('distance', np.float32), was included... is it still required? Hard to calculate
        ("normalized_x", np.float32),
        ("normalized_y", np.float32),
        ("normalized_z", np.float32),
        ("cell_sigma", np.float32),
        ("track_chi2_dof", np.float32),
        # ("track_chi2_dof_cell_sigma", np.float32), TBD if combined classes are required/useful
        ("cell_E", np.float32),
        ("normalized_cell_E", np.float32),
        ("track_pt", np.float32),
        ("normalized_track_pt", np.float32),
        # ('track_pt_cell_E', np.float32), TBD if combined classes are required/useful
        # ('normalized_track_pt_cell_E', np.float32), TBD if combined classes are required/useful
    ]
)


def event_to_trainable(
    event: ak.Record,
    focal_index: int,
    delta_r_max: float,
    max_event_len: int,
    truth=True,
) -> np.ndarray:
    trainable_array = []

    if ak.num(event["tracks"], axis=0) == 0:
        log.debug("No tracks contained in event, skipping")
        # TODO: throw error here, this should never be reached in production
        return np.zeros(max_event_len, dtype=event_array_dtype)

    # TODO: This process crashes if there are no track hits for the focal track

    focal_track = event["tracks"][focal_index]
    focal_eta = focal_track["trackEta"]
    focal_phi = focal_track["trackPhi"]

    # handle other tracks
    all_track_eta = event["tracks"]["trackEta"]
    all_track_phi = event["tracks"]["trackPhi"]
    delta_r = calculate_delta_r(
        eta1=focal_eta, phi1=focal_phi, eta2=all_track_eta, phi2=all_track_phi
    )
    track_indices = ak.local_index(all_track_eta)
    within_delta_r = (delta_r < delta_r_max) & (track_indices != focal_index)
    adjacent_tracks = event["tracks"][within_delta_r]

    # Handle cells within delta R
    cell_eta = event["cells"]["eta"]
    cell_phi = event["cells"]["phi"]
    delta_r_cells = calculate_delta_r(
        eta1=cell_eta, phi1=cell_phi, eta2=focal_eta, phi2=focal_phi
    )
    mask_cells = delta_r_cells < delta_r_max
    adjacent_cells = event["cells"][mask_cells]

    # run processing
    focal_data = add_focal_track(focal_track)
    track_data = add_non_focal_tracks(adjacent_tracks, focal_track)
    # TODO: sort by max E so that cutoff does not cause issues
    cell_data = add_cells(adjacent_cells, focal_track, truth=truth)

    # combine fields
    event_data = np.concatenate([focal_data, track_data, cell_data])
    event_data["runNumber"] = event["runNumber"]
    event_data["eventNumber"] = event["eventNumber"]

    event_data = truncate_or_pad(event_data, max_event_len)
    event_data = normalize_event_data(event_data)
    trainable_array.append(event_data)

    # Convert to numpy array
    trainable_array = np.stack(trainable_array)[0]
    return trainable_array


def normalize_event_data(event_data):
    # Normalize x, y, z
    positions = np.vstack((event_data["x"], event_data["y"], event_data["z"])).T
    norms = np.linalg.norm(positions, axis=1)
    norms[norms == 0] = 1  # Avoid division by zero

    # TODO: Should these be normalized together to scale the space or independently?
    event_data["normalized_x"] = event_data["x"] / norms
    event_data["normalized_y"] = event_data["y"] / norms
    event_data["normalized_z"] = event_data["z"] / norms

    # Normalize cell_E and track_pt
    cell_E_valid = event_data["cell_E"][event_data["cell_E"] != -1]
    if len(cell_E_valid) > 0:
        max_cell_E = np.max(cell_E_valid)
    else:
        max_cell_E = 1  # Avoid division by zero
    event_data["normalized_cell_E"] = np.where(
        event_data["cell_E"] != -1, event_data["cell_E"] / max_cell_E, -1
    )

    # For tracks
    track_pt_valid = event_data["track_pt"][event_data["track_pt"] != -1]
    if len(track_pt_valid) > 0:
        max_track_pt = np.max(track_pt_valid)
    else:
        max_track_pt = 1  # Avoid division by zero
    event_data["normalized_track_pt"] = np.where(
        event_data["track_pt"] != -1, event_data["track_pt"] / max_track_pt, -1
    )

    return event_data


def truncate_or_pad(event_data, max_event_len):
    current_len = len(event_data)
    if current_len > max_event_len:  # truncate
        event_data = event_data[:max_event_len]
    elif current_len < max_event_len:  # pad
        pad_length = max_event_len - current_len
        padding = np.zeros(pad_length, dtype=event_array_dtype)
        padding.fill(SENTINEL_NO_DATA)
        event_data = np.concatenate([event_data, padding])
    return event_data


def add_focal_track(track):
    n_hits = ak.num(track["hits"], axis=0)

    trainable = np.zeros(n_hits, dtype=event_array_dtype)
    trainable.fill(SENTINEL_NO_DATA)

    trainable["category"] = POINT_TYPE_ENCODING["focal_track"]

    chi2 = ak.to_numpy(track["trackChiSquared"])
    ndof = ak.to_numpy(track["trackNumberDOF"])
    chi2_by_ndof = chi2 / ndof
    track_index = track["track_index"]
    track_pt = ak.to_numpy(track["trackPt"])
    track_num = 0  # focal track takes 0 to show model that it is the focal track

    x = ak.to_numpy(track["hits"]["x"])
    y = ak.to_numpy(track["hits"]["y"])
    z = ak.to_numpy(track["hits"]["z"])
    eta = ak.to_numpy(track["hits"]["eta"])
    phi = ak.to_numpy(track["hits"]["phi"])

    trainable["x"] = x
    trainable["y"] = y
    trainable["z"] = z
    trainable["eta"] = eta
    trainable["phi"] = phi
    trainable["track_num"] = track_num
    trainable["track_chi2_dof"] = chi2_by_ndof
    trainable["track_pt"] = track_pt
    trainable["delta_r"] = 0
    trainable["track_index"] = track_index

    return trainable


def add_non_focal_tracks(tracks, focal_track):
    hits_per_track = ak.num(tracks["hits"])
    total_hits = ak.sum(hits_per_track)

    trainable = np.zeros(total_hits, dtype=event_array_dtype)
    trainable.fill(SENTINEL_NO_DATA)

    trainable["category"] = POINT_TYPE_ENCODING["non_focal_track"]

    # track data
    chi2 = ak.to_numpy(tracks["trackChiSquared"])
    ndof = ak.to_numpy(tracks["trackNumberDOF"])
    track_pt = ak.to_numpy(tracks["trackPt"])
    track_nums = np.arange(1, len(tracks) + 1)
    track_idxs = ak.to_numpy(tracks["track_index"])

    # repeat for array fitting
    chi2_by_ndof_per_hit = np.repeat(chi2 / ndof, hits_per_track)
    track_pt_per_hit = np.repeat(track_pt, hits_per_track)
    track_num_per_hit = np.repeat(track_nums, hits_per_track)
    track_idxs_per_hit = np.repeat(track_idxs, hits_per_track)

    # hit data
    x = ak.to_numpy(ak.flatten(tracks["hits"]["x"]))
    y = ak.to_numpy(ak.flatten(tracks["hits"]["y"]))
    z = ak.to_numpy(ak.flatten(tracks["hits"]["z"]))
    eta = ak.to_numpy(ak.flatten(tracks["hits"]["eta"]))
    phi = ak.to_numpy(ak.flatten(tracks["hits"]["phi"]))

    trainable["x"] = x
    trainable["y"] = y
    trainable["z"] = z
    trainable["eta"] = eta
    trainable["phi"] = phi
    trainable["track_num"] = track_num_per_hit
    trainable["track_chi2_dof"] = chi2_by_ndof_per_hit
    trainable["track_pt"] = track_pt_per_hit
    trainable["track_index"] = track_idxs_per_hit

    delta_r = calculate_delta_r(
        eta, focal_track["trackEta"], phi, focal_track["trackPhi"]
    )
    trainable["delta_r"] = delta_r

    return trainable


def add_cells(cells, focal_track, truth: bool):
    cell_data = np.zeros(ak.num(cells, axis=0), dtype=event_array_dtype)
    cell_data.fill(SENTINEL_NO_DATA)

    cell_data["category"] = POINT_TYPE_ENCODING["cell"]

    # Extract arrays
    x = ak.to_numpy(cells["x"])
    y = ak.to_numpy(cells["y"])
    z = ak.to_numpy(cells["z"])
    eta = ak.to_numpy(cells["eta"])
    phi = ak.to_numpy(cells["phi"])
    cell_sigma = ak.to_numpy(cells["cell_sigma"])
    cell_e = ak.to_numpy(cells["cell_E"])
    cell_id = ak.to_numpy(cells["cell_ID"])

    cell_data["cell_ID"] = cell_id
    cell_data["x"] = x
    cell_data["y"] = y
    cell_data["z"] = z
    cell_data["eta"] = eta
    cell_data["phi"] = phi
    cell_data["cell_sigma"] = cell_sigma
    cell_data["cell_E"] = cell_e

    delta_r = calculate_delta_r(
        eta, focal_track["trackEta"], phi, focal_track["trackPhi"]
    )
    cell_data["delta_r"] = delta_r

    if truth:
        cell_truth_total_e = ak.to_numpy(cells["cell_hitsTruthTotalE"])
        cell_data["truth_cell_total_energy"] = cell_truth_total_e

        focal_particle_idx = focal_track["trackTruthParticleIndex"]
        cell_hits_truth_index = cells["cell_hitsTruthIndex"]
        cell_hits_truth_e = cells["cell_hitsTruthE"]

        is_focal_particle = cell_hits_truth_index == focal_particle_idx
        focal_energy_per_cell = ak.sum(
            cell_hits_truth_e * is_focal_particle, axis=-1
        ).to_numpy()
        cell_data["truth_cell_focal_energy"] = focal_energy_per_cell

        cell_data["truth_cell_focal_fraction_energy"] = np.divide(
            focal_energy_per_cell,
            cell_truth_total_e,
            out=np.zeros_like(focal_energy_per_cell),
            where=(cell_truth_total_e != 0),
        )

        cell_data["truth_cell_focal_observed_fraction_energy"] = np.divide(
            focal_energy_per_cell,
            cell_e,
            out=np.zeros_like(focal_energy_per_cell),
            where=(cell_e != 0),
        )

    return cell_data
