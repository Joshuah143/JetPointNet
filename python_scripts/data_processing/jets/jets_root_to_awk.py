import uproot
import awkward as ak
import numpy as np
import glob
import time
from pathlib import Path
from multiprocessing import Pool
import os
from tqdm.auto import tqdm

import sys

REPO_PATH = Path.home() / "workspace/jetpointnet"
SCRIPT_PATH = REPO_PATH / "python_scripts"
sys.path.append(str(SCRIPT_PATH))


from data_processing.jets.preprocessing_header import *
from data_processing.jets.awk_utils import *
from data_processing.jets.common_utils import *

track_layer_branches = [f"trackEta_{layer}" for layer in calo_layers] + [
    f"trackPhi_{layer}" for layer in calo_layers
]
jets_other_included_fields = [
    "trackSubtractedCaloEnergy",
    "trackPt",
    "nTrack",
    "cluster_cell_ID",
    "trackNumberDOF",
    "trackChiSquared",
    "cluster_cell_E",
    "cluster_cell_hitsTruthIndex",
    "cluster_cell_hitsTruthE",
    "trackTruthParticleIndex",
    "eventNumber",
]

fields_list = track_layer_branches + jets_other_included_fields


with uproot.open(str(GEO_FILE_LOC) + ":CellGeo") as cellgeo:
    cell_ID_geo = cellgeo["cell_geo_ID"].array(library="np")[0]
    eta_geo = cellgeo["cell_geo_eta"].array(library="np")[0]
    phi_geo = cellgeo["cell_geo_phi"].array(library="np")[0]
    rPerp_geo = cellgeo["cell_geo_rPerp"].array(library="np")[0]


def split_and_save_to_disk(processed_data, base_filename, id_splits: dict, save_locations: dict):
    """
    Split the processed data into TRAIN, VAL, and TEST sets and save them to disk.
    """
    num_events = len(processed_data["eventNumber"])

    train_mask, val_mask, test_mask = (
        np.zeros(num_events).astype("bool"),
        np.zeros(num_events).astype("bool"),
        np.zeros(num_events).astype("bool"),
    )

    for idx, event_idx in enumerate(processed_data.eventNumber):
        if len(event_idx) == 0:
            continue
        if np.isin(event_idx[0], id_splits["train_ids"]):
            train_mask[idx] = True
            id_splits["train_ids"] = id_splits["train_ids"][id_splits["train_ids"] != event_idx[0]]
        elif np.isin(event_idx[0], id_splits["val_ids"]):
            val_mask[idx] = True
            id_splits["val_ids"] = id_splits["val_ids"][ id_splits["val_ids"] != event_idx[0]]
        else:
            test_mask[idx] = True
            id_splits["test_ids"] = id_splits["test_ids"][id_splits["test_ids"] != event_idx[0]]

    for mask, folder, split in zip(
        [train_mask, val_mask, test_mask],
        [save_locations["train_dir"], save_locations["val_dir"], save_locations["test_dir"]],
        ["train", "val", "test"],
    ):
        if not np.any(mask):
            print(f"No events in {split} split for this chunk!")
            continue
        data = processed_data[mask]
        ak.to_parquet(data, os.path.join(folder, base_filename.split("/")[-1] + f"_{split}.parquet"))


def setup_directories(dataset_name):
    # Directories for train, val, and test sets
    train_dir = os.path.join(AWK_SAVE_LOC(AWK), "train", dataset_name)
    val_dir = os.path.join(AWK_SAVE_LOC(AWK), "val", dataset_name)
    test_dir = os.path.join(AWK_SAVE_LOC(AWK), "test", dataset_name)

    # Ensure directories exist
    os.makedirs(AWK_SAVE_LOC(AWK), exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    return {"train_dir": train_dir, "val_dir": val_dir, "test_dir": test_dir}

def process_events(
    data,
    cell_ID_geo,
    cell_eta_geo,
    cell_phi_geo,
    cell_rPerp_geo,
    thread_id,
    # progress_dict,
):
    """
    Main event processing function.

    Parameters:
    - data: The events data to process.
    - cellgeo: The cell geometry data.
    """
    tracks_sample = ak.ArrayBuilder()  # Initialize the awkward array structure for track samples
    for event_idx, event in enumerate(data):
        if DEBUG_NUM_EVENTS_TO_USE is not None:
            if (
                event_idx >= DEBUG_NUM_EVENTS_TO_USE
            ):  # Limiting processing for demonstration
                break

        event_cells, event_cell_truths, track_etas, track_phis = (
            process_and_filter_cells(
                event, cell_ID_geo, cell_eta_geo, cell_phi_geo, cell_rPerp_geo
            )
        )  # Flatten and process cells in this event

        tracks_sample.begin_list()  # Start a new list for each event to hold tracks
        for track_idx in range(event["nTrack"]):
            # NOTE: this seem to work with ttbar 2k events dataset, however it may break if nTrack is not populated correctly in MC (this seems to be the case with old version of rho_full dataset)

            # Retrieve focal track's intersection points for distance calculation
            focal_track_intersections = calculate_track_intersections(
                {layer: eta[track_idx] for layer, eta in track_etas.items()},
                {layer: phi[track_idx] for layer, phi in track_phis.items()},
            )
            focal_points = [(x, y, z) for _, (x, y, z, eta, phi) in focal_track_intersections.items()]

            if len(focal_points) == 0:
                print(f"Track {track_idx} of event {event['eventNumber']} skipped due to no track hits")
                continue

            tracks_sample.begin_record()  # Each track is a record within the event list

            # Meta info
            fields = [
                ("eventNumber", "integer"),
                ("trackEta_EMB2", "real"),
                ("trackPhi_EMB2", "real"),
                ("trackEta_EME2", "real"),
                ("trackPhi_EME2", "real"),
                ("trackSubtractedCaloEnergy", "real"),
                ("trackPt", "real"),
                ("trackChiSquared/trackNumberDOF", "real"),
            ]
            track_eta_ref, track_phi_ref, track_part_Idx = add_track_meta_info(
                tracks_sample, event, event_idx, track_idx, fields
            )

            # Track intersections
            track_intersections = add_track_intersection_info(
                tracks_sample, track_idx, track_etas, track_phis
            )

            # Associated cell info
            process_associated_cell_info(
                event_cells,
                event_cell_truths,
                track_part_Idx,
                tracks_sample,
                track_eta_ref,
                track_phi_ref,
                track_intersections,
            )

            # Associated tracks
            process_associated_tracks(
                event,
                tracks_sample,
                track_eta_ref,
                track_phi_ref,
                track_idx,
                event["nTrack"],
                track_etas,
                track_phis,
                focal_points,
            )

            tracks_sample.end_record()  # End the record for the current track
            break # TEMP MUST REMOVE

        tracks_sample.end_list()  # End the list for the current event
        # progress_dict[str(thread_id)] = event_idx / len(data)
    return (
        tracks_sample.snapshot()
    )  # Convert the ArrayBuilder to an actual Awkward array and return it


def process_events_wrapper(args):
    return process_events(*args)


def process_chunk(chunk, cell_ID_geo, cell_eta_geo, cell_phi_geo, cell_rPerp_geo):
    """
    Modified `process_chunk` to handle progress reporting using multiprocessing.Pool and tqdm.
    """
    chunk_size = len(chunk)
    events_per_thread = chunk_size // AWK_THREADS_PER_CHUNK

    # Create a list of arguments for each thread
    args = [
        (
            chunk[start_idx:end_idx],
            cell_ID_geo,
            cell_eta_geo,
            cell_phi_geo,
            cell_rPerp_geo,
            i,
            # progress_dict,
        )
        for i in range(AWK_THREADS_PER_CHUNK)
        for start_idx, end_idx in [
            (
                i * events_per_thread,
                (
                    chunk_size
                    if i == AWK_THREADS_PER_CHUNK - 1
                    else (i + 1) * events_per_thread
                ),
            )
        ]
    ]

    with Pool(processes=AWK_THREADS_PER_CHUNK) as pool:
        results = list(tqdm(pool.imap(process_events_wrapper, args), total=len(args)))

    combined_array = ak.concatenate(results)
    return combined_array

def event_handler_wrapper(filepath):
    print(f"Handling {filepath}")

    data_set_name = prefix_to_set[filepath.split("/")[-1][:FILE_PREFIX_LEN]]
    save_locations = setup_directories(data_set_name)
    chunk_counter = 0

    base_filename = f"{filepath}_chunk_{chunk_counter}"

    root_filename = base_filename.split("/")[-1]
    
    existence_test_file = root_filename + f"_train.parquet"

    if not OVERWRITE_AWK:
        print(f"Testing for existence of: {os.path.join(save_locations['train_dir'], existence_test_file)}")

    # early return if the file already exists
    if not OVERWRITE_AWK and os.path.exists(os.path.join(save_locations['train_dir'], existence_test_file)):
        print(f"Already converted, skipping: {base_filename.split('/')[-1]}")
        return

    try:
        with uproot.open(filepath + ":EventTree") as events:
            id_split = split_data(
                events, split_seed=62, retrieve=False
            )
            
            for chunk in events.iterate(
                fields_list, library="ak", step_size=NUM_MAX_EVENTS_PER_CHUNK
            ):
                print(f"\nProcessing chunk {chunk_counter + 1} of size {len(chunk)}")

                processed_data = process_chunk(chunk, cell_ID_geo, eta_geo, phi_geo, rPerp_geo)
                base_filename = f"{filepath}_chunk_{chunk_counter}"
                split_and_save_to_disk(processed_data, base_filename, id_split, save_locations)

                chunk_counter += 1
    except uproot.exceptions.KeyInFileError:
        print(f"Skipping {filepath}, no EventTree found")


def main():
    # print("Events Keys:")
    # for key in events.keys():
    #     print(key)
    # print("\nGeometry Keys:")
    # for key in cellgeo.keys():
    #     print(key)
    # print()

    start_time = time.time()

    if os.path.isfile(ROOT_FILES_DIR):
        event_handler_wrapper(ROOT_FILES_DIR)
    else:
        potential_root_files = glob.glob(os.path.join(ROOT_FILES_DIR, "**/*.root"), recursive=True)
        root_files = [entry for entry in potential_root_files if os.path.isfile(entry)]
        number_of_files = len(root_files)
        for i, file in enumerate(root_files):
            event_handler_wrapper(file)
            print(f"Completed {i+1:<7} of {number_of_files}")
        

    end_time = time.time()
    print(f"Total Time Elapsed: {(end_time - start_time) / 60 / 60:.2f} Hours")


if __name__ == "__main__":
    main()
