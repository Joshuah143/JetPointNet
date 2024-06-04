import numpy as np
import awkward as ak
import sys
from data_processing.jets.track_metadata import (
    calo_layers,
    has_fixed_r,
    fixed_r,
    fixed_z,
)  # Assuming these are correctly defined

HAS_FIXED_R, FIXED_R, FIXED_Z = has_fixed_r, fixed_r, fixed_z
from data_processing.jets.preprocessing_header import *


POINT_TYPE_LABELS = {0: "focus hit", 1: "cell", 2: "unfocus hit", -1: "padding"}
POINT_TYPE_ENCODING = {v: k for k, v in POINT_TYPE_LABELS.items()}

# =======================================================================================================================
# ============ UTILITY FUNCTIONS ================================================================================


def calculate_cartesian_coordinates(eta, phi, rPerp):
    X = rPerp * np.cos(phi)
    Y = rPerp * np.sin(phi)
    Z = rPerp * np.sinh(eta)

    return X, Y, Z


def eta_phi_to_cartesian(eta, phi, R=1):
    # theta = 2 * np.arctan(np.exp(-eta))
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    z = R * np.sinh(eta)  # Corrected to use sinh
    return x, y, z


# Define the function to calculate the intersection with a fixed R layer
def intersection_fixed_r(eta, phi, fixed_r):
    x, y, z = eta_phi_to_cartesian(eta, phi, R=fixed_r)
    return x, y, z


# Define the function to calculate the intersection with a fixed Z layer
def intersection_fixed_z(eta, phi, fixed_z):
    x, y, z_unit = eta_phi_to_cartesian(eta, phi)
    scale_factor = fixed_z / z_unit
    x *= scale_factor
    y *= scale_factor
    z = fixed_z
    return x, y, z


# Helper function to calculate delta R using eta and phi directly
def calculate_delta_r(eta1, phi1, eta2, phi2):
    dphi = np.mod(phi2 - phi1 + np.pi, 2 * np.pi) - np.pi
    deta = eta2 - eta1
    return np.sqrt(deta**2 + dphi**2)


# =======================================================================================================================


# Define the function to calculate the intersection points for each track
def calculate_track_intersections(track_eta, track_phi):
    intersections = {}
    for layer in calo_layers:
        eta = track_eta[layer]
        phi = track_phi[layer]
        # Skip calculation for invalid eta, phi values
        if eta < -100000 or phi < -100000:
            continue

        # Calculate intersection based on layer type
        if HAS_FIXED_R.get(layer, False):
            x, y, z = intersection_fixed_r(eta, phi, FIXED_R[layer])
        elif layer in FIXED_Z:
            x, y, z = intersection_fixed_z(eta, phi, FIXED_Z[layer])
        else:
            raise Exception(
                "Error: cell layers must either be fixed R or fixed Z, and not neither"
            )
        intersections[layer] = (x, y, z, eta, phi)
    return intersections


# =======================================================================================================================


def calculate_max_sample_length(tracks_array):
    """Compute maximum number of points, plus keep track of point types per event and track"""
    n_points = np.zeros(shape=(len(ak.flatten(tracks_array)), 5))
    max_length = 0
    current_index = 0
    for event in tracks_array:
        for track_idx, track in enumerate(event):
            n_focus_track_hits = len(track["track_layer_intersections"])
            n_associated_cells_hits = len(track["associated_cells"])
            length = n_focus_track_hits + n_associated_cells_hits
            if len(track["associated_tracks"]) > 0:
                for associated_track in track["associated_tracks"]:
                    n_associated_track_hits = len(
                        associated_track["track_layer_intersections"]
                    )
                    n_points[current_index] = [
                        event.eventNumber[0],
                        track_idx,
                        n_focus_track_hits,
                        n_associated_cells_hits,
                        n_associated_track_hits,
                    ]
                    length += n_associated_track_hits
            else:
                n_points[current_index] = [
                    event.eventNumber[0],
                    track_idx,
                    n_focus_track_hits,
                    n_associated_cells_hits,
                    0,
                ]
            current_index += 1

            if length > max_length:
                max_length = length
    return max_length, n_points


# =======================================================================================================================


def print_events(tracks_sample_array, NUM_EVENTS_TO_PRINT):
    # Weirdly structured. Is overcomplicated code-wise for more readable output
    for event_idx, event in enumerate(ak.to_list(tracks_sample_array)):
        if event_idx >= NUM_EVENTS_TO_PRINT:
            break
        print("New event")
        # Each event can contain multiple tracks
        for track in event:
            # if (len(track["associated_cells"]) == 0):
            #    continue
            print("  Track")
            # Now, print each field and its value for the track
            for field in track:
                value = track[field]
                if field == "track_layer_intersections" or field == "associated_cells":
                    print(f"    {field}:")
                    for intpoint in value:
                        formatted_intpoint = {
                            k: (f"{v:.4f}" if isinstance(v, float) else v)
                            for k, v in intpoint.items()
                        }
                        print(f"        {formatted_intpoint}")
                elif field == "associated_tracks":
                    print(f"    {field}:")
                    for adj_track in value:
                        for adj_field in adj_track:
                            adj_value = adj_track[adj_field]
                            if adj_field == "track_layer_intersections":
                                print(f"            {adj_field}:")
                                for layer_point in adj_value:
                                    formatted_layer_point = {
                                        k: (f"{v:.4f}" if isinstance(v, float) else v)
                                        for k, v in layer_point.items()
                                    }
                                    print(f"                {formatted_layer_point}")
                            else:
                                if isinstance(adj_value, float):
                                    print(f"            {adj_field}: {adj_value:.4f}")
                                else:
                                    print(f"            {adj_field}: {adj_value}")
                else:
                    if isinstance(
                        value, float
                    ):  # Check if the value is a float and format it
                        print(f"    {field}: {value:.4f}")
                    else:  # If not a float, print the value as is
                        print(f"    {field}: {value}")
            print()


# =======================================================================================================================
def add_train_label_record(
        *, # ensures that all arguments must be named
        track_points: list,
        event_number: int,
        category: int | str,
        delta_R: float,
        normalized_x: float,
        normalized_y: float,
        normalized_z: float,
        normalized_distance: float,
        cell_E: float = -1,
        track_pt: float = -1,
        cell_ID: int = -1,
        track_ID: int = -1,
        track_num: int = -1,
):
    
    category = category if type(category) is int else POINT_TYPE_ENCODING[category]
    track_points.append(
                    (
                        event_number,
                        cell_ID,
                        track_ID,
                        delta_R,
                        ## above is only for traceability, should not be included in training data
                        category,
                        track_num, # used to associate non focal track interactions without track ID leakage
                        normalized_x,
                        normalized_y,
                        normalized_z,
                        normalized_distance,
                        cell_E,
                        track_pt
                    )
                )
    



def build_input_array(tracks_sample_array, max_sample_length, energy_scale=1):
    samples = []

    for event in tracks_sample_array:
        for track in event:
            if len(track["associated_cells"]) < 25:
                continue

            event_array = []

            # NOTE: I think this should be better moved to preprocessing at training time and done on whole training data rather than chunk-wise
            # Gather all track, cell, and associated track points to find min and max values for normalization
            all_points = []
            distances = []
            for intersection in track["track_layer_intersections"]:
                all_points.append(
                    (intersection["X"], intersection["Y"], intersection["Z"])
                )
            for cell in track["associated_cells"]:
                all_points.append((cell["X"], cell["Y"], cell["Z"]))
                distances.append(cell["distance_to_track"])
            for associated_track in track["associated_tracks"]:
                for intersection in associated_track["track_layer_intersections"]:
                    all_points.append(
                        (intersection["X"], intersection["Y"], intersection["Z"])
                    )
                    distances.append(intersection["distance_to_track"])

            # Calculate min and max for normalization
            min_x, min_y, min_z = np.min(all_points, axis=0)
            max_x, max_y, max_z = np.max(all_points, axis=0)
            max_distance = max(distances)

            range_x, range_y, range_z = abs(max_x - min_x), abs(max_y - min_y), abs(max_z - min_z)

            # Normalize and add points
            for intersection in track["track_layer_intersections"]:
                normalized_x = (intersection["X"] - min_x) / range_x
                normalized_y = (intersection["Y"] - min_y) / range_y
                normalized_z = (intersection["Z"] - min_z) / range_z
                add_train_label_record(
                    track_points=event_array,
                    event_number=track["eventNumber"],
                    track_ID=track["trackID"],
                    category=POINT_TYPE_ENCODING["focus hit"],
                    delta_R=calculate_delta_r(track["trackEta"], track["trackPhi"], intersection["eta"], intersection["phi"]),
                    normalized_x=normalized_x,
                    normalized_y=normalized_y,
                    normalized_z=normalized_z,
                    normalized_distance=0,
                    track_pt=track["trackPt"],
                    cell_E=-1,
                    cell_ID=-1,
                    track_num=0
                )

            for cell in track["associated_cells"]:
                normalized_x = (cell["X"] - min_x) / range_x
                normalized_y = (cell["Y"] - min_y) / range_y
                normalized_z = (cell["Z"] - min_z) / range_z
                normalized_distance = cell["distance_to_track"] / max_distance
                add_train_label_record(
                    track_points=event_array,
                    event_number=track["eventNumber"],
                    track_ID=-1,
                    category=POINT_TYPE_ENCODING["cell"],
                    delta_R=calculate_delta_r(track["trackEta"], track["trackPhi"], cell["eta"], cell["phi"]),
                    normalized_x=normalized_x,
                    normalized_y=normalized_y,
                    normalized_z=normalized_z,
                    normalized_distance=normalized_distance,
                    track_pt=-1,
                    cell_E=cell["E"] * energy_scale,
                    cell_ID=cell["ID"],
                    track_num=-1
                )

            for track_idx, associated_track in enumerate(track["associated_tracks"]):
                for intersection in associated_track["track_layer_intersections"]:
                    normalized_x = (intersection["X"] - min_x) / range_x
                    normalized_y = (intersection["Y"] - min_y) / range_y
                    normalized_z = (intersection["Z"] - min_z) / range_z
                    normalized_distance = (
                        intersection["distance_to_track"] / max_distance
                    )
                    add_train_label_record(
                        track_points=event_array,
                        event_number=track["eventNumber"],
                        track_ID=associated_track["trackId"],
                        category=POINT_TYPE_ENCODING["unfocus hit"],
                        delta_R=intersection["delta_R_adj"],
                        normalized_x=normalized_x,
                        normalized_y=normalized_y,
                        normalized_z=normalized_z,
                        normalized_distance=normalized_distance,
                        track_pt=associated_track["trackPt"],
                        cell_E=-1,
                        cell_ID=-1,
                        track_num=track_idx + 1
                    )

            # Now, the sample is truncated to max_sample_length before padding is considered
            event_array = event_array[:max_sample_length]

            # Pad with zeros and -1 for class identity if needed
            num_points = len(event_array)
            if num_points < max_sample_length:
                for _ in range(max_sample_length - num_points):
                    add_train_label_record(
                        track_points=event_array,
                        event_number=-1,
                        track_ID=-1,
                        category=POINT_TYPE_ENCODING["padding"],
                        delta_R=-1,
                        normalized_x=-1,
                        normalized_y=-1,
                        normalized_z=-1,
                        normalized_distance=-1,
                        track_pt=-1,
                        cell_E=-1,
                        cell_ID=-1,
                        track_num=-1
                    )
            
    # I think this should gte saved as a tensorflow dataset, we could save them all as tensorflow datasets to be honest, merge them into one large file and let tensorflow handel the memory issues of that
            event_array_dtype = np.dtype([ # none can be unsigned because -1 is used as a pad for all, see above
                ('event_number', np.int32), # NOTE: there is a bug in tensorflow, all must conform to the same type
                ('cell_ID', np.int32),
                ('track_ID', np.int32),
                ('delta_R', np.float32),             
                ('category', np.int8),
                ('track_num', np.int32),
                ('normalized_x', np.float32),
                ('normalized_y', np.float32),
                ('normalized_z', np.float32),
                ('normalized_distance', np.float32),
                ('cell_E', np.float32),
                ('track_pt', np.float32),
            ])
            event_array_np = np.array(event_array, dtype=event_array_dtype)
            # Replace NaN values with 0
            event_array_np = np.nan_to_num(event_array_np, nan=0.0)
            samples.append(event_array_np)


    samples_array = np.array(samples)

    # samples_array = np.array(samples, dtype=np.float32)

    samples_array = np.nan_to_num(samples_array, nan=0.0)

    return samples_array


# =======================================================================================================================


def build_labels_array(
    tracks_sample_array, max_sample_length, label_string, label_scale=1
):
    labels_list = []

    for event in tracks_sample_array:
        for track in event:
            if len(track["associated_cells"]) < 25:
                continue

            label_array = np.full(max_sample_length, -1, dtype=np.float32)

            num_focused_track_points = len(track["track_layer_intersections"])
            num_associated_cells = len(track["associated_cells"])
            num_associated_track_points = sum(
                len(assoc_track["track_layer_intersections"])
                for assoc_track in track["associated_tracks"]
            )

            total_points = (
                num_focused_track_points
                + num_associated_cells
                + num_associated_track_points
            )
            total_points = min(
                total_points, max_sample_length
            )  # Ensure it doesn't exceed max_sample_length

            if add_tracks_as_labels == True:
                label_array[:num_focused_track_points] = 1.0
            else:
                label_array[:num_focused_track_points] = -1.0

            # Adjust for possible truncation
            end_cell_idx = min(
                num_focused_track_points + num_associated_cells, max_sample_length
            )
            label_array[num_focused_track_points:end_cell_idx] = (
                track["associated_cells"][label_string][
                    : end_cell_idx - num_focused_track_points
                ]
                * label_scale
            )

            start_idx = num_focused_track_points + num_associated_cells
            end_idx = start_idx + num_associated_track_points
            end_idx = min(end_idx, max_sample_length)  # Truncate if necessary

            if add_tracks_as_labels == True:
                label_array[start_idx:end_idx] = 0.0
            else:
                label_array[start_idx:end_idx] = -1.0

            labels_list.append(label_array)

    labels_array = np.array(labels_list, dtype=np.float32)

    # Replace NaN values with 0
    labels_array = np.nan_to_num(labels_array, nan=0.0)

    return labels_array


# =======================================================================================================================
# =======================================================================================================================


# =======================================================================================================================
# ============ PROCESSING CODE FUNCTIONS ================================================================================


def process_and_filter_cells(
    event, cell_ID_geo, cell_eta_geo, cell_phi_geo, cell_rPerp_geo
):
    """
    Parameters:
    - event: The event data containing cell and track information.
    - cellgeo: Geometric information about the cells.

    Returns:
    - event_cells: A structured array of filtered cells with added Cartesian coordinates.
    - track_etas: A dictionary of eta values for each track, organized by layer.
    - track_phis: A dictionary of phi values for each track, organized by layer.
    """

    # L: I don't think this has any impact. 1) may need to filter out empty cluster_ID elements; 2) the actual dimension is determined by the filtered cell_IDs later anyway;
    # NOTE: running without step 1 seems to produce same results
    # Extracting cell IDs and energies, assuming they are part of clusters

    # TODO use hitsE_EM for trucating so that it matches Jessica's code
    truncated_hitsTruthIndex = [
        cluster_hitsTruthIndex[: len(cluster_ID)]
        for cluster_ID, cluster_hitsTruthIndex in zip(
            event["cluster_cell_ID"], event["cluster_cell_hitsTruthIndex"]
        )
    ]
    truncated_hitsTruthE = [
        cluster_hitsTruthE[: len(cluster_ID)]
        for cluster_ID, cluster_hitsTruthE in zip(
            event["cluster_cell_ID"], event["cluster_cell_hitsTruthE"]
        )
    ]

    # Step 2: Flatten the arrays now that they've been truncated
    cell_IDs_with_multiples = ak.flatten(event["cluster_cell_ID"])
    cell_Es_with_multiples = ak.flatten(event["cluster_cell_E"])
    cell_part_truth_Idxs_with_multiples = ak.flatten(truncated_hitsTruthIndex)
    cell_part_truth_Es_with_multiples = ak.flatten(truncated_hitsTruthE)

    # print(len(cell_Es_with_multiples))
    # print(len(cell_IDs_with_multiples))
    # print(len(cell_part_truth_Es_with_multiples))
    # print(len(cell_part_truth_Idxs_with_multiples))

    # Finding unique cell IDs and their first occurrence indices
    _, unique_indices = np.unique(
        ak.to_numpy(cell_IDs_with_multiples), return_index=True
    )

    # Selecting corresponding unique cell data
    cell_IDs = cell_IDs_with_multiples[unique_indices]
    cell_Es = cell_Es_with_multiples[unique_indices]
    cell_hitsTruthIndices = cell_part_truth_Idxs_with_multiples[unique_indices]
    cell_hitsTruthEs = cell_part_truth_Es_with_multiples[unique_indices]

    # Matching cells with their geometric data
    cell_ID_geo_array = (
        cell_ID_geo  # np.array(cellgeo["cell_geo_ID"].array(library="ak")[0])
    )

    mask = np.isin(cell_ID_geo_array, np.array(cell_IDs))

    indices = np.where(mask)[0]

    # Extracting and mapping geometric data to the filtered cells
    cell_Etas = cell_eta_geo[
        indices
    ]  # cellgeo["cell_geo_eta"].array(library="ak")[0][indices]
    cell_Phis = cell_phi_geo[
        indices
    ]  # cellgeo["cell_geo_phi"].array(library="ak")[0][indices]
    cell_rPerps = cell_rPerp_geo[
        indices
    ]  # cellgeo["cell_geo_rPerp"].array(library="ak")[0][indices]

    # Calculating Cartesian coordinates for the cells
    cell_Xs, cell_Ys, cell_Zs = calculate_cartesian_coordinates(
        cell_Etas, cell_Phis, cell_rPerps
    )
    # Creating a structured array for the event's cells

    event_cells = ak.zip(
        {
            "ID": cell_IDs,
            "E": cell_Es,
            "eta": cell_Etas,
            "phi": cell_Phis,
            "X": cell_Xs,
            "Y": cell_Ys,
            "Z": cell_Zs,
        }
    )

    event_cell_truths = ak.zip(
        {
            "cell_hitsTruthIndices": cell_hitsTruthIndices,
            "cell_hitsTruthEs": cell_hitsTruthEs,
        }
    )

    # Preparing track eta and phi data for all layers
    track_etas = {layer: event[f"trackEta_{layer}"] for layer in calo_layers}
    track_phis = {layer: event[f"trackPhi_{layer}"] for layer in calo_layers}

    return event_cells, event_cell_truths, track_etas, track_phis


# =======================================================================================================================


def add_track_meta_info(tracks_sample, event, event_idx, track_idx, fields):
    """
    Adds track metadata information to the tracks_sample ArrayBuilder.

    Parameters:
    - tracks_sample: The Awkward ArrayBuilder to which the track metadata will be added.
    - event: The current event data containing track and other information.
    - track_idx: Index of the current track being processed.
    - fields: A list of tuples containing the field names and their types to be added.
    """
    # Start adding trackID, trackEta, and trackPhi as done previously
    tracks_sample.field("trackID").integer(track_idx)
    track_eta_ref = (
        event["trackEta_EMB2"][track_idx]
        if event["trackEta_EMB2"][track_idx] > UPROOT_MASK_VALUE_THRESHOLD
        else event["trackEta_EME2"][track_idx]
    )
    track_phi_ref = (
        event["trackPhi_EMB2"][track_idx]
        if event["trackPhi_EMB2"][track_idx] > UPROOT_MASK_VALUE_THRESHOLD
        else event["trackPhi_EME2"][track_idx]
    )
    tracks_sample.field("trackEta").real(track_eta_ref)
    tracks_sample.field("trackPhi").real(track_phi_ref)

    track_part_Idx = event["trackTruthParticleIndex"][track_idx]
    tracks_sample.field("track_part_Idx").integer(track_part_Idx)

    # Process additional fields based on the provided list
    for field_name, field_type in fields:
        tracks_sample.field(field_name)
        if field_type == "integer":
            # For integer fields
            if field_name == "eventNumber":
                tracks_sample.integer(event["eventNumber"])
            else:
                tracks_sample.integer(event[field_name][track_idx])
        elif field_type == "real":
            # For real number fields
            if field_name == "trackChiSquared/trackNumberDOF":
                tracks_sample.real(
                    event["trackChiSquared"][track_idx]
                    / event["trackNumberDOF"][track_idx]
                )

            elif not event[field_name][track_idx] < UPROOT_MASK_VALUE_THRESHOLD:
                tracks_sample.real(event[field_name][track_idx])

    return track_eta_ref, track_phi_ref, track_part_Idx


# =======================================================================================================================


def add_track_intersection_info(tracks_sample, track_idx, track_eta, track_phi):
    """
    Adds track X, Y, Z path points (intersections with cell layers) to the tracks_sample ArrayBuilder.

    Parameters:
    - tracks_sample: The Awkward ArrayBuilder to which the intersection points will be added.
    - track_idx: Index of the current track being processed.
    - track_eta: Dictionary of track eta values for each layer.
    - track_phi: Dictionary of track phi values for each layer.
    - calculate_track_intersections: Function to calculate the intersections of the track with cell layers.
    """
    # Calculate intersections for the track
    track_intersections = calculate_track_intersections(
        {layer: eta[track_idx] for layer, eta in track_eta.items()},
        {layer: phi[track_idx] for layer, phi in track_phi.items()},
    )



    # Add track intersection information
    tracks_sample.field("track_layer_intersections")
    tracks_sample.begin_list()  # Start list of intersection points for this track
    for layer, (x, y, z, eta, phi) in track_intersections.items():
        tracks_sample.begin_record()  # Each intersection point is a record
        tracks_sample.field("layer")
        tracks_sample.string(layer)
        tracks_sample.field("X")
        tracks_sample.real(x)
        tracks_sample.field("Y")
        tracks_sample.real(y)
        tracks_sample.field("Z")
        tracks_sample.real(z)
        tracks_sample.field("eta")
        tracks_sample.real(eta)
        tracks_sample.field("phi")
        tracks_sample.real(phi)
        tracks_sample.field("Label").real(1)
        tracks_sample.end_record()  # End the record for this intersection point
    tracks_sample.end_list()  # End list of intersection points

    return track_intersections


# =======================================================================================================================


def process_associated_cell_info(
    event_cells,
    event_cell_truths,
    track_part_Idx,
    tracks_sample,
    track_eta_ref,
    track_phi_ref,
    track_intersections,
):
    """
    Process cells associated with a track based on ΔR and other criteria.

    Parameters:
    - event_cells: The cells in the current event.
    - tracks_sample: Awkward ArrayBuilder for building the event structure.
    - track_eta_ref: Reference eta for the track.
    - track_phi_ref: Reference phi for the track.
    - track_intersections: Intersection points of the track with cell layers.
    - MAX_DISTANCE: Maximum ΔR distance for a cell to be considered associated with the track.
    """

    # Use cell eta and phi directly from the `cells` structured array
    cell_eta = event_cells["eta"]
    cell_phi = event_cells["phi"]

    # Vectorized calculation of delta R for all cells with respect to the track
    delta_r = calculate_delta_r(track_eta_ref, track_phi_ref, cell_eta, cell_phi)

    # Creating a mask for cells within the delta R threshold
    mask = delta_r <= MAX_DISTANCE

    # Apply the mask to filter cells directly using Awkward Array's boolean masking
    filtered_cells = event_cells[mask]
    filtered_cell_truths = event_cell_truths[mask]

    tracks_sample.field("total_associated_cell_energy").real(
        ak.sum(filtered_cells["E"])
    )

    # Preparing to add the filtered cells to the track sample
    tracks_sample.field("associated_cells")
    tracks_sample.begin_list()

    track_intersection_points = [
        (x, y, z) for layer, (x, y, z, eta, phi) in track_intersections.items()
    ]
    # NOTE: same as above but easier
    # list(track_intersections.values())

    for cell_idx in range(len(filtered_cells)):
        # TODO: MAKE IT ONLY ADD CELLS THAT HAVE ANY TRUTH_HIT_INDEX IN THEM
        tracks_sample.begin_record()
        tracks_sample.field("ID").integer(filtered_cells[cell_idx]["ID"])
        tracks_sample.field("E").real(filtered_cells[cell_idx]["E"])
        tracks_sample.field("X").real(filtered_cells[cell_idx]["X"])
        tracks_sample.field("Y").real(filtered_cells[cell_idx]["Y"])
        tracks_sample.field("Z").real(filtered_cells[cell_idx]["Z"])

        # Calculate distances to each track intersection point and find the minimum
        cell_x, cell_y, cell_z = (
            filtered_cells[cell_idx]["X"],
            filtered_cells[cell_idx]["Y"],
            filtered_cells[cell_idx]["Z"],
        )
        # NOTE: the following should reproduce min_dist calculation but more readable (even faster?)
        # np.sqrt(np.array(track_intersection_points) - np.array([cell_x, cell_y, cell_z]) )

        # dists = [np.sqrt((x - cell_x) ** 2 + (y - cell_y) ** 2 + (z - cell_z) ** 2)
        #     for x, y, z in track_intersection_points]
        # np.linalg.norm(np.array(track_intersection_points)[0] - np.array([cell_x, cell_y, cell_z]))

        # a_min_b = np.array(track_intersection_points) - np.array([cell_x, cell_y, cell_z])
        # min(np.sqrt(np.einsum('ij,ij->i', a_min_b, a_min_b)))

        min_distance = min(
            np.sqrt((x - cell_x) ** 2 + (y - cell_y) ** 2 + (z - cell_z) ** 2)
            for x, y, z in track_intersection_points
        )
        tracks_sample.field("distance_to_track").real(min_distance)
        tracks_sample.field("eta").real(filtered_cells[cell_idx]["eta"])
        tracks_sample.field("phi").real(filtered_cells[cell_idx]["phi"])

        cell_part_IDs = filtered_cell_truths[cell_idx]["cell_hitsTruthIndices"]
        cell_part_Es = filtered_cell_truths[cell_idx]["cell_hitsTruthEs"]

        total_energy = np.sum(
            filtered_cell_truths[cell_idx]["cell_hitsTruthEs"]
        )  # Sum of all particle energy deposits in the cell
        tracks_sample.field("Total_Truth_Energy").real(total_energy)

        if track_part_Idx in cell_part_IDs:
            found_index = np.where(cell_part_IDs == track_part_Idx)[0][
                0
            ]  # Locate index of track_part_Idx in cell_part_IDs
            part_energy = filtered_cell_truths[cell_idx]["cell_hitsTruthEs"][
                found_index
            ]  # Retrieve corresponding energy deposit
            energy_fraction = part_energy / total_energy  # Calculate energy fraction
            # L: NOTE should we refer to measured rather than truth energy instead?
            tracks_sample.field("Fraction_Label").real(energy_fraction)
            tracks_sample.field("Total_Label").real(total_energy)
        else:
            tracks_sample.field("Fraction_Label").real(0)
            tracks_sample.field("Total_Label").real(0)

        tracks_sample.field("cell_Hits_TruthIndices")
        tracks_sample.begin_list()
        for part in cell_part_IDs:
            tracks_sample.integer(part)
        tracks_sample.end_list()

        tracks_sample.field("cell_Hits_TruthEs")
        tracks_sample.begin_list()
        for part in cell_part_Es:
            tracks_sample.real(part)
        tracks_sample.end_list()
        # print(filtered_cell_truths[cell_idx]["cell_hitsTruthIndices"])
        # print(filtered_cell_truths[cell_idx]["cell_hitsTruthEs"])

        tracks_sample.end_record()

    tracks_sample.end_list()


# =======================================================================================================================


def process_associated_tracks(
    event,
    tracks_sample,
    track_eta_ref,
    track_phi_ref,
    track_idx,
    nTrack,
    track_etas,
    track_phis,
):
    """
    Process tracks associated with a focal track based on ΔR and other criteria.

    Parameters:
    - event: The current event data including track information.
    - tracks_sample: Awkward ArrayBuilder for building the event structure.
    - track_eta_ref: Reference eta for the focal track.
    - track_phi_ref: Reference phi for the focal track.
    - track_idx: Index of the focal track within the event.
    - nTrack: Total number of tracks in the event.
    - MAX_DISTANCE: Maximum ΔR distance for a track to be considered associated with the focal track.
    - calculate_delta_r: Function to calculate the delta R between two points.
    - calculate_track_intersections: Function to calculate track intersections with cell layers.
    - UPROOT_MASK_VALUE_THRESHOLD: Threshold value to determine valid track points.
    """

    # Initialize the field for adjacent tracks
    tracks_sample.field("associated_tracks")
    tracks_sample.begin_list()

    # Retrieve focal track's intersection points for distance calculation
    focal_track_intersections = calculate_track_intersections(
        {layer: eta[track_idx] for layer, eta in track_etas.items()},
        {layer: phi[track_idx] for layer, phi in track_phis.items()},
    )
    focal_points = [(x, y, z) for _, (x, y, z, eta, phi) in focal_track_intersections.items()]

    # Iterate over all tracks in the event to find adjacent tracks
    for adj_track_idx in range(nTrack):
        if adj_track_idx == track_idx:  # Skip the focal track itself
            continue

        # Determine reference eta/phi for the adjacent track
        adj_track_eta = (
            event["trackEta_EMB2"][adj_track_idx]
            if event["trackEta_EMB2"][adj_track_idx] > UPROOT_MASK_VALUE_THRESHOLD
            else event["trackEta_EME2"][adj_track_idx]
        )
        adj_track_phi = (
            event["trackPhi_EMB2"][adj_track_idx]
            if event["trackPhi_EMB2"][adj_track_idx] > UPROOT_MASK_VALUE_THRESHOLD
            else event["trackPhi_EME2"][adj_track_idx]
        )

        # Calculate delta R between focal and adjacent track
        delta_r_adj = calculate_delta_r(
            track_eta_ref, track_phi_ref, adj_track_eta, adj_track_phi
        )

        # Check if adjacent track is within MAX_DISTANCE
        if delta_r_adj <= MAX_DISTANCE:
            tracks_sample.begin_record()
            tracks_sample.field("trackId").integer(adj_track_idx)
            tracks_sample.field("track_part_Idx").integer(
                event["trackTruthParticleIndex"][adj_track_idx]
            )
            tracks_sample.field("trackPt").real(event["trackPt"][adj_track_idx])

            tracks_sample.field("track_layer_intersections")
            tracks_sample.begin_list()
            adj_track_intersections = calculate_track_intersections(
                {layer: eta[adj_track_idx] for layer, eta in track_etas.items()},
                {layer: phi[adj_track_idx] for layer, phi in track_phis.items()},
            )

            for layer, (x, y, z, eta, phi) in adj_track_intersections.items():
                min_distance_to_focal = min(
                    np.sqrt((fx - x) ** 2 + (fy - y) ** 2 + (fz - z) ** 2)
                    for fx, fy, fz in focal_points
                )

                tracks_sample.begin_record()
                tracks_sample.field("layer").string(layer)
                tracks_sample.field("X").real(x)
                tracks_sample.field("Y").real(y)
                tracks_sample.field("Z").real(z)
                tracks_sample.field("delta_R_adj").real(delta_r_adj) # same for all layers
                tracks_sample.field("distance_to_track").real(min_distance_to_focal)
                tracks_sample.field("Label").real(0)
                tracks_sample.end_record()

            tracks_sample.end_list()
            tracks_sample.end_record()

    tracks_sample.end_list()


# =======================================================================================================================
# =======================================================================================================================


# ============ train/val/test split: this works on eventNumber as index to subset data ================================================================================


def train_val_test_split_events(
    all_events_ids,
    train_pct=TRAIN_SPLIT_RATIO,
    val_pct=VAL_SPLIT_RATIO,
    split_seed=None,
):
    from sklearn.model_selection import train_test_split

    if not split_seed:
        split_seed = np.random.choice(range(100), size=1)[0]
    train_ids, val_ids = train_test_split(
        all_events_ids, test_size=1 - train_pct, random_state=split_seed
    )
    val_ids, test_ids = train_test_split(
        val_ids, test_size=1 - (val_pct) / (1 - train_pct), random_state=split_seed
    )

    for event_ids, fn in zip([train_ids, val_ids, test_ids], ["train", "val", "test"]):
        AWK_SAVE_LOC.mkdir(exist_ok=True, parents=True)
        np.savetxt(
            AWK_SAVE_LOC.parent / f"{fn}_events_{split_seed=}.txt", event_ids, fmt="%d"
        )

    return split_seed, train_ids, val_ids, test_ids


def get_split(split_seed):
    ids = []
    for split in ["train", "val", "test"]:
        ids.append(
            np.loadtxt(AWK_SAVE_LOC.parent.parent / "AwkwardArrs" / f"{split}_events_{split_seed=}.txt")
        )
    return split_seed, *ids


def split(events, split_seed):
    all_events_ids = events["eventNumber"].array(library="np")
    split_seed, train_ids, val_ids, test_ids = train_val_test_split_events(
        all_events_ids, split_seed=split_seed
    )
    return split_seed, train_ids, val_ids, test_ids


def split_data(events, split_seed, retrieve=True):
    if retrieve:
        return get_split(split_seed)
    else:
        return split(events, split_seed)


# =======================================================================================================================#
