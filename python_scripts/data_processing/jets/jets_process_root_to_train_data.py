import uproot
import awkward as ak
import numpy as np
import time
import concurrent.futures
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from multiprocessing import Manager, Pool


from both import *
from util_functs import *


manager = Manager()
progress_dict = manager.dict()


def process_events_with_progress(data, cell_ID_geo, cell_eta_geo, cell_phi_geo, cell_rPerp_geo, thread_id, progress_dict):
    """
    Wrapper for the original `process_events` function that updates progress.
    """
    progress_dict[str(thread_id)] = 0.0  # Initialize progress
    result = process_events(data, cell_ID_geo, cell_eta_geo, cell_phi_geo, cell_rPerp_geo)
    progress_dict[str(thread_id)] = 1.0  # Update progress to 100% when done
    return result


def process_events(data, cell_ID_geo, cell_eta_geo, cell_phi_geo, cell_rPerp_geo):
    """
    Main event processing function.
    
    Parameters:
    - data: The events data to process.
    - cellgeo: The cell geometry data.
    """
    tracks_sample = ak.ArrayBuilder()  # Initialize the awkward array structure for track samples
    for event_idx, event in enumerate(data):
        if DEBUG_NUM_EVENTS_TO_USE is not None:
            if event_idx >= DEBUG_NUM_EVENTS_TO_USE:  # Limiting processing for demonstration
                break

        
        event_cells, event_cell_truths, track_etas, track_phis = process_and_filter_cells(event, cell_ID_geo, cell_eta_geo, cell_phi_geo, cell_rPerp_geo)

        
        tracks_sample.begin_list()  # Start a new list for each event to hold tracks
        for track_idx in range(event["nTrack"]):
            
            tracks_sample.begin_record()  # Each track is a record within the event list

            # Meta info
            fields = [
                ("eventID", "integer"),
                ("trackEta_EMB2", "real"),
                ("trackPhi_EMB2", "real"),
                ("trackEta_EME2", "real"),
                ("trackPhi_EME2", "real"),
                ("trackSubtractedCaloEnergy", "real"),
                ("trackPt", "real"),
                ("trackChiSquared/trackNumberDOF", "real"),
            ]
            track_eta_ref, track_phi_ref, track_part_Idx = add_track_meta_info(tracks_sample, event, event_idx, track_idx, fields)

            # Track intersections
            track_intersections = add_track_intersection_info(tracks_sample, track_idx, track_etas, track_phis)

            # Associated cell info
            process_associated_cell_info(event_cells, event_cell_truths, track_part_Idx, tracks_sample, track_eta_ref, track_phi_ref, track_intersections)

            # Associated tracks
            process_associated_tracks(event, tracks_sample, track_eta_ref, track_phi_ref, track_idx, event["nTrack"], track_etas, track_phis)

            tracks_sample.end_record()  # End the record for the current track

        tracks_sample.end_list()  # End the list for the current event
    return tracks_sample.snapshot()  # Convert the ArrayBuilder to an actual Awkward array and return it

def save_to_disk(processed_data, filename):
    """
    Save the processed data to disk.
    """
    # Example: saving as a Parquet file (implementation depends on the desired format)
    ak.to_parquet(processed_data, filename)

def process_chunk_with_progress(chunk, cell_ID_geo, cell_eta_geo, cell_phi_geo, cell_rPerp_geo, progress_dict):
    """
    Modified `process_chunk` to handle progress reporting using multiprocessing.Pool.
    """
    chunk_size = len(chunk)
    events_per_thread = chunk_size // NUM_THREAD_PER_CHUNK
    args = [(chunk[start_idx:end_idx], cell_ID_geo, cell_eta_geo, cell_phi_geo, cell_rPerp_geo, i, progress_dict)
            for i in range(NUM_THREAD_PER_CHUNK)
            for start_idx, end_idx in [(i * events_per_thread, chunk_size if i == NUM_THREAD_PER_CHUNK - 1 else (i + 1) * events_per_thread)]]

    with Pool(processes=NUM_THREAD_PER_CHUNK) as pool:
        results = pool.starmap(process_events_with_progress, args)

    combined_array = ak.concatenate(results)
    return combined_array

def monitor_progress(progress_dict):
    """
    Monitor and print the progress of each thread in real-time.
    """
    while True:
        progresses = [f"Th{i+1}: {progress_dict.get(str(i), 0):.1%}" for i in range(NUM_THREAD_PER_CHUNK)]
        print('\r' + ', '.join(progresses), end="", flush=True)
        if all(progress_dict.get(str(i), 0) == 1.0 for i in range(NUM_THREAD_PER_CHUNK)):
            break
        time.sleep(0.1)  # Update interval (in seconds)
    print("\nAll threads completed.")

if __name__ == "__main__":
    # Initialize the Manager and the shared dictionary for tracking progress
    manager = Manager()
    progress_dict = manager.dict()

    # Set up a thread to monitor progress and start it
    progress_thread = threading.Thread(target=monitor_progress, args=(progress_dict,))
    progress_thread.start()

    events = uproot.open(FILE_LOC + ":EventTree")
    cellgeo = uproot.open(GEO_FILE_LOC + ":CellGeo")

    # Your existing setup code remains unchanged
    print("Events Keys:")
    for key in events.keys():
        print(key)
    print("\nGeometry Keys:")
    for key in cellgeo.keys():
        print(key)
    print()

    track_layer_branches = [f'trackEta_{layer}' for layer in calo_layers] + [f'trackPhi_{layer}' for layer in calo_layers]
    jets_other_included_fields = ["trackSubtractedCaloEnergy", "trackPt", "nTrack", "cluster_cell_ID",
                                  "trackNumberDOF", "trackChiSquared", "cluster_cell_E", "cluster_cell_hitsTruthIndex", "cluster_cell_hitsTruthE", "trackTruthParticleIndex"]
    fields_list = track_layer_branches + jets_other_included_fields

    cell_ID_geo = cellgeo["cell_geo_ID"].array(library="np")[0]
    eta_geo = cellgeo["cell_geo_eta"].array(library="np")[0]
    phi_geo = cellgeo["cell_geo_phi"].array(library="np")[0]
    rPerp_geo = cellgeo["cell_geo_rPerp"].array(library="np")[0]

    start_time = time.time()
    chunk_counter = 0
    for chunk in events.iterate(fields_list, library="ak", step_size=NUM_EVENTS_PER_CHUNK):
        print(f"\nProcessing chunk {chunk_counter + 1} of size {len(chunk)}")
        # Initialize the progress dictionary for the new chunk
        for i in range(NUM_THREAD_PER_CHUNK):
            progress_dict[str(i)] = 0.0

        processed_data = process_chunk_with_progress(chunk, cell_ID_geo, eta_geo, phi_geo, rPerp_geo, progress_dict)
        filename = f"{SAVE_LOC}processed_chunk_{chunk_counter}.parquet"
        save_to_disk(processed_data, filename)
        chunk_counter += 1

    # Ensure the monitoring thread knows processing is complete
    for i in range(NUM_THREAD_PER_CHUNK):
        progress_dict[str(i)] = 1.0

    # Wait for the monitoring thread to finish
    progress_thread.join()

    end_time = time.time()
    print("Total Time Elapsed: ", (end_time - start_time) / 60 / 60, " Hours")