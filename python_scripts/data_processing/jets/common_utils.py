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


def calculate_max_sample_length_simplified(tracks_array):
    """Compute maximum number of points"""
    current_index = 0
    track_points_arr = []
    cells_in_track_arr = []
    number_of_adj_arr = []
    for event in tracks_array:
        for track in event:
            n_focus_track_hits = len(track["track_layer_intersections"])
            n_associated_cells_hits = len(track["associated_cells"])
            length = n_focus_track_hits + n_associated_cells_hits
            if len(track["associated_tracks"]) > 0:
                for associated_track in track["associated_tracks"]:
                    n_associated_track_hits = len(
                        associated_track["track_layer_intersections"]
                    )
                    length += n_associated_track_hits
    
            current_index += 1
            track_points_arr.append(length)
            cells_in_track_arr.append(len(track["associated_cells"]))
            number_of_adj_arr.append(len(track["associated_tracks"]))
    return track_points_arr, cells_in_track_arr, number_of_adj_arr


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
