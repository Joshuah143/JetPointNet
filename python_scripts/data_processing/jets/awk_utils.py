import numpy as np
import awkward as ak
import sys
from particle.pdgid import charge
from data_processing.jets.track_metadata import (
    calo_layers,
    has_fixed_r,
    fixed_r,
    fixed_z,
)  # Assuming these are correctly defined

HAS_FIXED_R, FIXED_R, FIXED_Z = has_fixed_r, fixed_r, fixed_z
from data_processing.jets.preprocessing_header import *
from data_processing.jets.common_utils import (calculate_cartesian_coordinates,
                                               calculate_delta_r, 
                                               intersection_fixed_r, 
                                               intersection_fixed_z)


# =======================================================================================================================
# ============ PROCESSING CODE FUNCTIONS ================================================================================


def process_and_filter_cells(
    event, cell_ID_geo, cell_eta_geo, cell_phi_geo, cell_rPerp_geo, cell_Sigma_geo
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
    # truncated_hitsTruthIndex = [
    #     cluster_hitsTruthIndex[: len(cluster_ID)]
    #     for cluster_ID, cluster_hitsTruthIndex in zip(
    #         event["cluster_cell_ID"], event["cluster_cell_hitsTruthIndex"]
    #     )
    # ]
    # truncated_hitsTruthE = [
    #     cluster_hitsTruthE[: len(cluster_ID)]
    #     for cluster_ID, cluster_hitsTruthE in zip(
    #         event["cluster_cell_ID"], event["cluster_cell_hitsTruthE"]
    #     )
    # ]

    # Step 2: Flatten the arrays now that they've been truncated
    cell_IDs_with_multiples_np = ak.to_numpy(ak.flatten(event["cluster_cell_ID"]))
    cell_Es_with_multiples_np = ak.to_numpy(ak.flatten(event["cluster_cell_E"]))
    cell_part_truth_Idxs_with_multiples = ak.flatten(event["cluster_cell_hitsTruthIndex"])
    cell_part_truth_Es_with_multiples = ak.flatten(event["cluster_cell_hitsTruthE"])

    # print(len(cell_Es_with_multiples))
    # print(len(cell_IDs_with_multiples))
    # print(len(cell_part_truth_Es_with_multiples))
    # print(len(cell_part_truth_Idxs_with_multiples))

    # SUM THE CELL_E ACROSS TOPO CLUSTERS


    # Rather than only take the unique cell_IDs, I want to summ the Cell_E for douplicate elements so that their energy is included. I also want to check for all doupliacte IDs that the truth indexes and Energies also match.

    unique_cell_IDs, unique_indices, cell_ID_counts = np.unique(cell_IDs_with_multiples_np, return_index=True, return_counts=True) 
    
    # Selecting corresponding unique cell data
    unique_cell_Es = cell_Es_with_multiples_np[unique_indices]
    unique_cell_hitsTruthIndices = cell_part_truth_Idxs_with_multiples[unique_indices]
    unique_cell_hitsTruthEs = cell_part_truth_Es_with_multiples[unique_indices]

    # Sum the cell_E values for each unique cell_ID
    for cell_id in enumerate(unique_cell_IDs[cell_ID_counts > 1]):
        unique_cell_Es[unique_cell_IDs == cell_id] += np.sum(cell_Es_with_multiples_np[cell_IDs_with_multiples_np == cell_id])


    # Matching cells with their geometric data
    cell_ID_geo_array = (
        cell_ID_geo  # np.array(cellgeo["cell_geo_ID"].array(library="ak")[0])
    )

    mask = np.isin(cell_ID_geo_array, np.array(unique_cell_IDs))

    indices = np.where(mask)[0]

    # Extracting and mapping geometric data to the filtered cells
    cell_Etas = cell_eta_geo[indices]  # cellgeo["cell_geo_eta"].array(library="ak")[0][indices]
    cell_Phis = cell_phi_geo[indices]  # cellgeo["cell_geo_phi"].array(library="ak")[0][indices]
    cell_rPerps = cell_rPerp_geo[indices]  # cellgeo["cell_geo_rPerp"].array(library="ak")[0][indices]
    cell_Sigmas = cell_Sigma_geo[indices]

    # Calculating Cartesian coordinates for the cells
    cell_Xs, cell_Ys, cell_Zs = calculate_cartesian_coordinates(
        cell_Etas, cell_Phis, cell_rPerps
    )
    # Creating a structured array for the event's cells

    event_cells = ak.zip(
        {
            "ID": unique_cell_IDs,
            "E": unique_cell_Es,
            "eta": cell_Etas,
            "phi": cell_Phis,
            "X": cell_Xs,
            "Y": cell_Ys,
            "Z": cell_Zs,
            "Sigma": cell_Sigmas
        }
    )

    truthPDGID_arr = event["truthPartPdgId"]

    cell_hitsTruthPDGIDs = [[truthPDGID_arr[index] for index in cell] for cell in unique_cell_hitsTruthIndices]
    cell_hitsTruthCharge = [[charge(pdgId) for pdgId in cell] for cell in cell_hitsTruthPDGIDs]

    event_cell_truths = ak.zip(
        {
            "cell_hitsTruthIndices": unique_cell_hitsTruthIndices,
            "cell_hitsTruthPDGIDs": cell_hitsTruthPDGIDs,
            "cell_hitsTruthCharges": cell_hitsTruthCharge,
            "cell_hitsTruthEs": unique_cell_hitsTruthEs,
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
        tracks_sample.field("Cell_Sigma").real(filtered_cells[cell_idx]["Sigma"])

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
        cell_part_charges = filtered_cell_truths[cell_idx]["cell_hitsTruthCharges"]
        cell_part_pdgIDs = filtered_cell_truths[cell_idx]["cell_hitsTruthPDGIDs"]

        # NOTE: This should be updated to the actual truth energy so th
        total_energy = np.sum(cell_part_Es)  # Sum of all particle energy deposits in the cell
        tracks_sample.field("Total_Truth_Energy").real(total_energy)

        focal_E = 0
        non_focal_E = 0
        nuetral_energy = 0
        for cell_part_ID, cell_truth_part_E, cell_truth_part_charge in zip(cell_part_IDs, cell_part_Es, cell_part_charges):
            if cell_part_ID == track_part_Idx:
                focal_E += cell_truth_part_E
            elif cell_truth_part_charge == 0:
                nuetral_energy += cell_truth_part_E
            else:
                non_focal_E += cell_truth_part_E

        tracks_sample.field("Focal_Fraction_Label").real(focal_E)
        tracks_sample.field("Non_Focal_Fraction_Label").real(non_focal_E)
        tracks_sample.field("Nuetral_Fraction_Label").real(nuetral_energy)
                
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

        tracks_sample.field("cell_Hits_TruthPDGIDs")
        tracks_sample.begin_list()
        for part in cell_part_pdgIDs:
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
    focal_points,
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
            chi2_dof = event["trackChiSquared"][adj_track_idx] / event["trackNumberDOF"][adj_track_idx]
            tracks_sample.field("trackChiSquared/trackNumberDOF").real(chi2_dof)
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
                tracks_sample.end_record()
            tracks_sample.end_list()
            tracks_sample.end_record()

    tracks_sample.end_list()


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
        AWK_SAVE_LOC(AWK).mkdir(exist_ok=True, parents=True)
        np.savetxt(
            AWK_SAVE_LOC(AWK).parent / f"{fn}_events_{split_seed=}.txt", event_ids, fmt="%d"
        )

    return split_seed, train_ids, val_ids, test_ids


def get_split(split_seed):
    ids = []
    for split in ["train", "val", "test"]:
        ids.append(
            np.loadtxt(AWK_SAVE_LOC(AWK).parent.parent / "AwkwardArrs" / f"{split}_events_{split_seed=}.txt")
        )
    return ids


def split(events, split_seed):
    all_events_ids = events["eventNumber"].array(library="np")
    split_seed, train_ids, val_ids, test_ids = train_val_test_split_events(
        all_events_ids, split_seed=split_seed
    )
    return {"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids}


def split_data(events, split_seed, retrieve=True):
    if retrieve:
        return get_split(split_seed)
    else:
        return split(events, split_seed)

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
# =======================================================================================================================
