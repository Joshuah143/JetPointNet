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
        truth_cell_fraction_energy: float,
        truth_cell_total_energy: float,
        normalized_x: float,
        normalized_y: float,
        normalized_z: float,
        normalized_distance: float,
        chi2_dof: float,
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
                        truth_cell_fraction_energy,
                        truth_cell_total_energy,
                        ## above is for the y values of the 
                        category,
                        track_num, # used to associate non focal track interactions without track ID leakage
                        normalized_x,
                        normalized_y,
                        normalized_z,
                        normalized_distance,
                        chi2_dof,
                        cell_E,
                        track_pt
                    )
                )
    

def build_input_array(tracks_sample_array, max_sample_length, energy_scale=1, include_chi2_dof=False):
    samples = []

    for event in tracks_sample_array:
        for track in event:
            if len(track["associated_cells"]) < 25:
                continue

            track_array = []

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
                    track_points=track_array,
                    event_number=track["eventNumber"],
                    track_ID=track["trackID"],
                    category=POINT_TYPE_ENCODING["focus hit"],
                    delta_R=calculate_delta_r(track["trackEta"], track["trackPhi"], intersection["eta"], intersection["phi"]),
                    truth_cell_fraction_energy=-1,
                    truth_cell_total_energy=-1,
                    normalized_x=normalized_x,
                    normalized_y=normalized_y,
                    normalized_z=normalized_z,
                    normalized_distance=0,
                    track_pt=track["trackPt"],
                    chi2_dof=track["trackChiSquared/trackNumberDOF"],
                    cell_E=-1,
                    cell_ID=-1,
                    track_num=0
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
                        track_points=track_array,
                        event_number=track["eventNumber"],
                        track_ID=associated_track["trackId"],
                        category=POINT_TYPE_ENCODING["unfocus hit"],
                        delta_R=intersection["delta_R_adj"],
                        truth_cell_fraction_energy=-1,
                        truth_cell_total_energy=-1,
                        normalized_x=normalized_x,
                        normalized_y=normalized_y,
                        normalized_z=normalized_z,
                        normalized_distance=normalized_distance,
                        chi2_dof=associated_track["trackChiSquared/trackNumberDOF"] if include_chi2_dof else -1,
                        track_pt=associated_track["trackPt"],
                        cell_E=-1,
                        cell_ID=-1,
                        track_num=track_idx + 1
                    )

            # sort by delta R so that if the event gets cut far cells will get cut first
            for cell in sorted(track["associated_cells"], key=lambda cell: calculate_delta_r(track["trackEta"], track["trackPhi"], cell["eta"], cell["phi"])):
                normalized_x = (cell["X"] - min_x) / range_x
                normalized_y = (cell["Y"] - min_y) / range_y
                normalized_z = (cell["Z"] - min_z) / range_z
                normalized_distance = cell["distance_to_track"] / max_distance
                add_train_label_record(
                    track_points=track_array,
                    event_number=track["eventNumber"],
                    track_ID=-1,
                    category=POINT_TYPE_ENCODING["cell"],
                    delta_R=calculate_delta_r(track["trackEta"], track["trackPhi"], cell["eta"], cell["phi"]),
                    truth_cell_fraction_energy=cell["Fraction_Label"],
                    truth_cell_total_energy=cell["Total_Truth_Energy"],
                    normalized_x=normalized_x,
                    normalized_y=normalized_y,
                    normalized_z=normalized_z,
                    normalized_distance=normalized_distance,
                    chi2_dof=-1,
                    track_pt=-1,
                    cell_E=cell["E"] * energy_scale,
                    cell_ID=cell["ID"],
                    track_num=-1
                )

            # Now, the sample is truncated to max_sample_length before padding is considered
            track_array = track_array[:max_sample_length]

            # Pad with zeros and -1 for class identity if needed
            num_points = len(track_array)
            if num_points < max_sample_length:
                for _ in range(max_sample_length - num_points):
                    add_train_label_record(
                        track_points=track_array,
                        event_number=-1,
                        track_ID=-1,
                        category=POINT_TYPE_ENCODING["padding"],
                        delta_R=-1,
                        truth_cell_fraction_energy=-1,
                        truth_cell_total_energy=-1,
                        normalized_x=-1,
                        normalized_y=-1,
                        normalized_z=-1,
                        normalized_distance=-1,
                        chi2_dof=-1,
                        track_pt=-1,
                        cell_E=-1,
                        cell_ID=-1,
                        track_num=-1
                    )
            
            event_array_dtype = np.dtype([ # none can be unsigned because -1 is used as a pad for all, see above
                ('event_number', np.int32),
                ('cell_ID', np.int32),
                ('track_ID', np.int32),
                ('delta_R', np.float32),
                ('truth_cell_fraction_energy', np.float32),
                ('truth_cell_total_energy', np.float32),
                ('category', np.int8),
                ('track_num', np.int32),
                ('normalized_x', np.float32),
                ('normalized_y', np.float32),
                ('normalized_z', np.float32),
                ('normalized_distance', np.float32),
                ("chi2_dof", np.float32),
                ('cell_E', np.float32),
                ('track_pt', np.float32),
            ])
            track_array_np = np.array(track_array, dtype=event_array_dtype)
            # Replace NaN values with 0
            track_array_np = np.nan_to_num(track_array_np, nan=0.0)
            samples.append(track_array_np)


    samples_array = np.array(samples)

    samples_array = np.nan_to_num(samples_array, nan=0.0)

    return samples_array


# =======================================================================================================================


# DEPRECATED - MOVED TO THE INPUT ARRAY
"""
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

    return labels_array"""
