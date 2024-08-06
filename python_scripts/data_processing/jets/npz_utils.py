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
from data_processing.jets.common_utils import calculate_delta_r

def add_train_label_record(
        *, # ensures that all arguments must be named
        track_points: list,
        event_number: int,
        category: int | str,
        delta_R: float,
        truth_cell_focal_fraction_energy: float,
        truth_cell_non_focal_fraction_energy: float,
        truth_cell_neutral_fraction_energy: float,
        truth_cell_total_energy: float,
        x: float,
        y: float,
        z: float,
        distance: float,
        normalized_x: float,
        normalized_y: float,
        normalized_z: float,
        normalized_distance: float,
        track_chi2_dof: float,
        cell_sigma: float,
        cell_E: float = NPZ_PAD_VAL,
        normalized_cell_E = NPZ_PAD_VAL,
        track_pt: float = NPZ_PAD_VAL,
        normalized_track_pt: float = NPZ_PAD_VAL,
        cell_ID: int = NPZ_PAD_VAL,
        track_ID: int = NPZ_PAD_VAL,
        track_num: int = NPZ_PAD_VAL,
):
    
    category = category if type(category) is int else POINT_TYPE_ENCODING[category]
    track_points.append( # NOTE: Must match the dtype for the npz array
                    (
                        event_number,
                        cell_ID,
                        track_ID,
                        delta_R,
                        ## above is only for traceability, should not be included in training data
                        truth_cell_focal_fraction_energy,
                        truth_cell_non_focal_fraction_energy,
                        truth_cell_neutral_fraction_energy,
                        truth_cell_total_energy,
                        ## above is for the y values of the 
                        category,
                        track_num, # used to associate non focal track interactions without track ID leakage
                        x,
                        y,
                        z,
                        distance,
                        normalized_x,
                        normalized_y,
                        normalized_z,
                        normalized_distance,
                        cell_sigma,
                        track_chi2_dof,
                        cell_sigma if category == POINT_TYPE_ENCODING['cell'] else track_chi2_dof, 
                        cell_E,
                        normalized_cell_E,
                        track_pt,
                        normalized_track_pt,
                        cell_E if category == POINT_TYPE_ENCODING['cell'] else track_pt, 
                        normalized_cell_E if category == POINT_TYPE_ENCODING['cell'] else normalized_track_pt, 
                    )
                )
    

def add_multitrack_truths(
    cells_list,
    track_partID_to_idx,
    cell_Hits_TruthIndices, 
    cell_Hits_TruthRelitiveEnergy
    ):
    track_effects = [0] * MAX_TRACK_ASSOCIATIONS
    for truthIdx, relitive_E in zip(cell_Hits_TruthIndices, cell_Hits_TruthRelitiveEnergy):
        if truthIdx in track_partID_to_idx.keys():
            index = track_partID_to_idx[truthIdx]
            if index < MAX_TRACK_ASSOCIATIONS:
                track_effects[index] = relitive_E
    
    cells_list.append(track_effects)


def build_input_array(tracks_sample_array, max_sample_length):
    samples = []
    cell_hits_truths = []

    for event in tracks_sample_array:
        for sample in event:
            if len(sample["associated_cells"]) < MIN_TRACK_CELL_HITS:
                continue

            if sample['track_part_Idx'] == NPZ_PAD_VAL or NPZ_PAD_VAL in sample['associated_tracks']['track_part_Idx']:
                continue # Skip clusters with 'fake' tracks

            track_array = []
            cells_list = []

            # NOTE: I think this should be better moved to preprocessing at training time and done on whole training data rather than chunk-wise
            # Gather all track, cell, and associated track points to find min and max values for normalization
            all_points = []
            distances = []
            for intersection in sample["track_layer_intersections"]:
                all_points.append(
                    (intersection["X"], intersection["Y"], intersection["Z"])
                )
            for cell in sample["associated_cells"]:
                all_points.append((cell["X"], cell["Y"], cell["Z"]))
                distances.append(cell["distance_to_track"])
            for associated_track in sample["associated_tracks"]:
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
            for intersection in sample["track_layer_intersections"]:
                normalized_x = (intersection["X"] - min_x) / range_x
                normalized_y = (intersection["Y"] - min_y) / range_y
                normalized_z = (intersection["Z"] - min_z) / range_z
                add_train_label_record(
                    track_points=track_array,
                    event_number=sample["eventNumber"],
                    track_ID=sample["trackID"],
                    category=POINT_TYPE_ENCODING["focus hit"],
                    delta_R=calculate_delta_r(sample["trackEta"], sample["trackPhi"], intersection["eta"], intersection["phi"]),
                    truth_cell_focal_fraction_energy=NPZ_PAD_VAL,
                    truth_cell_non_focal_fraction_energy=NPZ_PAD_VAL,
                    truth_cell_neutral_fraction_energy=NPZ_PAD_VAL,
                    truth_cell_total_energy=NPZ_PAD_VAL,
                    x=intersection["X"],
                    y=intersection["Y"],
                    z=intersection["Z"],
                    distance=0,
                    normalized_x=normalized_x,
                    normalized_y=normalized_y,
                    normalized_z=normalized_z,
                    normalized_distance=0,
                    track_pt=sample["trackPt"],
                    normalized_track_pt=sample["trackPt"]/sample['total_sample_track_pt'],
                    track_chi2_dof=sample["trackChiSquared/trackNumberDOF"],
                    cell_sigma=NPZ_PAD_VAL,
                    cell_E=NPZ_PAD_VAL,
                    normalized_cell_E=NPZ_PAD_VAL,
                    cell_ID=NPZ_PAD_VAL,
                    track_num=0
                )
                add_multitrack_truths(cells_list, {}, [], [])


            track_partID_to_idx = {sample["track_part_Idx"]: 0}
            for track_idx, associated_track in enumerate(sorted(sample["associated_tracks"], key=lambda adj_track: adj_track['trackPt'])):
                track_partID_to_idx[associated_track['track_part_Idx']] = track_idx + 1
                for intersection in associated_track["track_layer_intersections"]:
                    normalized_x = (intersection["X"] - min_x) / range_x
                    normalized_y = (intersection["Y"] - min_y) / range_y
                    normalized_z = (intersection["Z"] - min_z) / range_z
                    normalized_distance = (
                        intersection["distance_to_track"] / max_distance
                    )
                    add_train_label_record(
                        track_points=track_array,
                        event_number=sample["eventNumber"],
                        track_ID=associated_track["trackId"],
                        category=POINT_TYPE_ENCODING["unfocus hit"],
                        delta_R=intersection["delta_R_adj"],
                        truth_cell_focal_fraction_energy=NPZ_PAD_VAL,
                        truth_cell_non_focal_fraction_energy=NPZ_PAD_VAL,
                        truth_cell_neutral_fraction_energy=NPZ_PAD_VAL,
                        truth_cell_total_energy=NPZ_PAD_VAL,
                        x=intersection["X"],
                        y=intersection["Y"],
                        z=intersection["Z"],
                        distance=intersection["distance_to_track"],
                        normalized_x=normalized_x,
                        normalized_y=normalized_y,
                        normalized_z=normalized_z,
                        normalized_distance=normalized_distance,
                        track_chi2_dof=associated_track["trackChiSquared/trackNumberDOF"],
                        cell_sigma=NPZ_PAD_VAL,
                        track_pt=associated_track["trackPt"],
                        normalized_track_pt=associated_track["trackPt"]/sample['total_sample_track_pt'],
                        cell_E=NPZ_PAD_VAL,
                        normalized_cell_E=NPZ_PAD_VAL,
                        cell_ID=NPZ_PAD_VAL,
                        track_num=track_idx + 1
                    )
                    add_multitrack_truths(cells_list, {}, [], [])


            # sort by delta R so that if the event gets cut far cells will get cut first
            for cell in sorted(sample["associated_cells"], key=lambda cell: calculate_delta_r(sample["trackEta"], sample["trackPhi"], cell["eta"], cell["phi"])):
                normalized_x = (cell["X"] - min_x) / range_x
                normalized_y = (cell["Y"] - min_y) / range_y
                normalized_z = (cell["Z"] - min_z) / range_z
                normalized_distance = cell["distance_to_track"] / max_distance
                add_train_label_record(
                    track_points=track_array,
                    event_number=sample["eventNumber"],
                    track_ID=NPZ_PAD_VAL,
                    category=POINT_TYPE_ENCODING["cell"],
                    delta_R=calculate_delta_r(sample["trackEta"], sample["trackPhi"], cell["eta"], cell["phi"]),
                    truth_cell_focal_fraction_energy=cell["Focal_Fraction_Label"],
                    truth_cell_non_focal_fraction_energy=cell["Non_Focal_Fraction_Label"],
                    truth_cell_neutral_fraction_energy=cell["Nuetral_Fraction_Label"],
                    truth_cell_total_energy=cell["Total_Truth_Energy"],
                    x=cell["X"],
                    y=cell["Y"],
                    z=cell["Z"],
                    distance=cell["distance_to_track"],
                    normalized_x=normalized_x,
                    normalized_y=normalized_y,
                    normalized_z=normalized_z,
                    normalized_distance=normalized_distance,
                    track_chi2_dof=NPZ_PAD_VAL,
                    cell_sigma=cell['Cell_Sigma'],
                    track_pt=NPZ_PAD_VAL,
                    normalized_track_pt=NPZ_PAD_VAL,
                    cell_E=cell["E"],
                    normalized_cell_E=cell["E"]/sample['total_associated_cell_energy'],
                    cell_ID=cell["ID"],
                    track_num=NPZ_PAD_VAL
                )
                add_multitrack_truths(cells_list, track_partID_to_idx, cell["cell_Hits_TruthIndices"], cell["cell_Hits_TruthEs"]/cell["E"])


            # Now, the sample is truncated to max_sample_length before padding is considered
            track_array = track_array[:max_sample_length]

            # Pad with zeros and NPZ_PAD_VAL for class identity if needed
            num_points = len(track_array)
            if num_points < max_sample_length:
                for _ in range(max_sample_length - num_points):
                    add_train_label_record(
                        track_points=track_array,
                        event_number=NPZ_PAD_VAL,
                        track_ID=NPZ_PAD_VAL,
                        category=POINT_TYPE_ENCODING["padding"],
                        delta_R=NPZ_PAD_VAL,
                        truth_cell_focal_fraction_energy=NPZ_PAD_VAL,
                        truth_cell_non_focal_fraction_energy=NPZ_PAD_VAL,
                        truth_cell_neutral_fraction_energy=NPZ_PAD_VAL,
                        truth_cell_total_energy=NPZ_PAD_VAL,
                        x=NPZ_PAD_VAL,
                        y=NPZ_PAD_VAL,
                        z=NPZ_PAD_VAL,
                        distance=NPZ_PAD_VAL,
                        normalized_x=NPZ_PAD_VAL,
                        normalized_y=NPZ_PAD_VAL,
                        normalized_z=NPZ_PAD_VAL,
                        normalized_distance=NPZ_PAD_VAL,
                        track_chi2_dof=NPZ_PAD_VAL,
                        cell_sigma=NPZ_PAD_VAL,
                        track_pt=NPZ_PAD_VAL,
                        normalized_track_pt=NPZ_PAD_VAL,
                        cell_E=NPZ_PAD_VAL,
                        normalized_cell_E=NPZ_PAD_VAL,
                        cell_ID=NPZ_PAD_VAL,
                        track_num=NPZ_PAD_VAL
                    )
                    add_multitrack_truths(cells_list, {}, [], [])

            
            event_array_dtype = np.dtype([ # none can be unsigned because NPZ_PAD_VAL is used as a pad for all, see above
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
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('distance', np.float32),
                ('normalized_x', np.float32),
                ('normalized_y', np.float32),
                ('normalized_z', np.float32),
                ('normalized_distance', np.float32),
                ('cell_sigma', np.float32),
                ('track_chi2_dof', np.float32),
                ("track_chi2_dof_cell_sigma", np.float32),
                ('cell_E', np.float32),
                ('normalized_cell_E', np.float32),
                ('track_pt', np.float32),
                ('normalized_track_pt', np.float32),
                ('track_pt_cell_E', np.float32),
                ('normalized_track_pt_cell_E', np.float32),
            ])
            track_array_np = np.array(track_array, dtype=event_array_dtype)
            # Replace NaN values with NPZ_PAD_VAL
            track_array_np = np.nan_to_num(track_array_np, nan=NPZ_PAD_VAL)
            samples.append(track_array_np)
            cell_hits_truths.append(cells_list)

    samples_array = np.array(samples)

    samples_array = np.nan_to_num(samples_array, nan=0.0)

    return samples_array, cell_hits_truths
