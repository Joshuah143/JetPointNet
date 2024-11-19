from pathlib import Path
from typing import List, Dict, Any
import uproot
import awkward as ak
from multiprocessing import Pool
import numpy as np
import os
from tqdm import tqdm
from itertools import repeat

from coordinate_conversions import intersection_fixed_z, eta_phi_to_cartesian
from track_metadata import fixed_r, fixed_z

NUM_THREADS = os.cpu_count()
EVENT_BATCHES = 20

def load_from_root(root_files_location: Path, geo_file=Path("data/rho_small.root"), truth: bool=True) -> ak.Array:
    with uproot.open(geo_file)["CellGeo"] as geo_locations:
        calo_geo = {"ID": geo_locations['cell_geo_ID'].array()[0],
                    "eta": geo_locations['cell_geo_eta'].array()[0],
                    "phi": geo_locations['cell_geo_phi'].array()[0],
                    "rPerp": geo_locations['cell_geo_rPerp'].array()[0],
                    "sigma": geo_locations['cell_geo_sigma'].array()[0]}

    events_iterator = uproot.iterate({root_files_location: "EventTree"}, step_size=EVENT_BATCHES)
    # TODO: save intermediates to prevent lost work
    with Pool(processes=NUM_THREADS) as pool:
        processed_events = list(pool.starmap(
            build_awk_arr,
            zip(filtered_events_iterator(events_iterator), repeat(calo_geo), repeat(truth))
        ))

    ak_arr_batched = ak.Array(processed_events)
    ak_arr = ak.flatten(ak_arr_batched)
    return ak_arr

    # events = next(uproot.iterate({root_files_location: "EventTree"}, step_size=EVENT_BATCHES))
    # processed_events = [build_awk_arr(events, calo_geo, truth)]
    # return ak.Array([item for sublist in processed_events for item in sublist])

def filter_events(event_batch):
    INVALID_TRUTH_INDEX = -1

    valid_tracks = ak.all(event_batch['trackTruthParticleIndex'] != INVALID_TRUTH_INDEX, axis=-1)
    valid_clusters = ak.all(
        ak.all(
            ak.all(event_batch['cluster_cell_hitsTruthIndex'] != INVALID_TRUTH_INDEX, axis=-1),
            axis=-1
        ),
        axis=-1
    )

    valid_events_mask = valid_tracks & valid_clusters
    return event_batch[valid_events_mask]

def filtered_events_iterator(events_iterator):
    for event_batch in events_iterator:
        filtered_batch = filter_events(event_batch)
        if len(filtered_batch) > 0:
            yield filtered_batch


def build_awk_arr(event_tree: ak.highlevel.Array, geo_dict: dict, truth=True) -> ak.Array:
    return ak.Array([process_event(event, geo_dict, truth) for event in event_tree])

def process_event(event: ak.Record, geo_dict: dict, truth=True) -> dict:
    updated_event = {"runNumber": event["runNumber"],
                     "eventNumber": event["eventNumber"],
                     "lumiBlock": event["lumiBlock"],
                     "coreFlags": event["coreFlags"],
                     "tracks": generate_tracks(event, truth),
                     "cells": generate_cells(event, geo_dict, truth),
                     "attributed": {[]}}

    return updated_event

def generate_tracks(event, truth=True) -> list[dict[str, list[dict[str, Any] | dict[str, Any]] | Any]]:
    tracks = []
    for idx in range(event['nTrack']):
        track = {
            'track_index': idx,
            'trackPt': event['trackPt'][idx],
            'trackP': event['trackP'][idx],
            'trackMass': event['trackMass'][idx],
            'trackEta': event['trackEta'][idx],
            'trackPhi': event['trackPhi'][idx],
            'trackChiSquared': event['trackChiSquared'][idx],
            'trackNumberDOF': event['trackNumberDOF'][idx],
            'trackD0': event['trackD0'][idx],
            'trackZ0': event['trackZ0'][idx],
        }

        if truth:
            truth_part_idx = event['trackTruthParticleIndex'][idx]
            track_truth = {
                'trackTruthParticleIndex': truth_part_idx,
                'truthPartStatus': event['truthPartStatus'][truth_part_idx],
                'truthPartPt': event['truthPartPt'][truth_part_idx],
                'truthPartE': event['truthPartE'][truth_part_idx],
                'truthPartMass': event['truthPartMass'][truth_part_idx],
                'truthPartEta': event['truthPartEta'][truth_part_idx],
                'truthPartPhi': event['truthPartPhi'][truth_part_idx],
                'truthPartBarcode': event['truthPartBarcode'][truth_part_idx],
                'truthPartPdgId': event['truthPartPdgId'][truth_part_idx],
            }
            track.update(track_truth)

        calo_hits = []

        for layer, radius in fixed_r.items():
            eta = ak.to_numpy(event[f"trackEta_{layer}"])[idx]
            phi = ak.to_numpy(event[f"trackPhi_{layer}"])[idx]
            if phi > -9999999: # Default value for N/A data
                x, y, z = eta_phi_to_cartesian(eta, phi, radius)
                hit = {
                    'x': x,
                    'y': y,
                    'z': z,
                    'eta': eta,
                    'phi': phi,
                }
                calo_hits.append(hit)
        for layer, z in fixed_z.items():
            eta = ak.to_numpy(event[f"trackEta_{layer}"])[idx]
            phi = ak.to_numpy(event[f"trackPhi_{layer}"])[idx]
            if phi > -9999999:
                x, y, _ = intersection_fixed_z(eta, phi, z)
                hit = {
                    'x': x,
                    'y': y,
                    'z': z,
                    'eta': eta,
                    'phi': phi,
                }
                calo_hits.append(hit)

        track['hits'] = calo_hits
        tracks.append(track)
    return tracks

def generate_cells(event, geo_dict, truth=True) -> list:
    cells = []

    for cluster_idx in range(event['nCluster']):
        # NOTE: event['cluster_nCells'][cluster_idx] cannot be since it does not have 5 MeV cell cut
        for cell_idx in range(len(event['cluster_cell_ID'][cluster_idx])):
            cell_ID = event['cluster_cell_ID'][cluster_idx][cell_idx]

            if len(idx_list := np.where(geo_dict["ID"] == cell_ID)) == 0:
                print(f"CELL_ID: {cell_ID} does not exist in geo file")
                continue

            idx = idx_list[0]
            eta = geo_dict["eta"][idx]
            phi = geo_dict["phi"][idx]
            rPerp = geo_dict["rPerp"][idx]
            x, y, z = eta_phi_to_cartesian(eta, phi, rPerp)

            # TODO: Combine duplicate cells, some cells are split between clusters
            cell = {
                "cell_cluster_index": cluster_idx,
                "cell_E": event['cluster_cell_E'][cluster_idx][cell_idx],
                "cell_ID": cell_ID,
                "cell_sigma": geo_dict["sigma"][idx][0],
                'eta': eta[0],
                'phi': phi[0],
                'x': x[0],
                'y': y[0],
                'z': z[0],
            }

            if truth:
                cell_truth = {
                    "cell_hitsTruthIndex": event['cluster_cell_hitsTruthIndex'][cluster_idx][cell_idx],
                    'cell_hitsTruthE': event['cluster_cell_hitsTruthE'][cluster_idx][cell_idx],
                    'cell_hitsTruthTotalE': event['cluster_cell_hitsTruthTotalE'][cluster_idx][cell_idx],
                }
                cell.update(cell_truth)
            cells.append(cell)
    return cells


if __name__ == "__main__":
    loaded = load_from_root("data/JZ4/*.root")
    with open("test_input_JZ4.json", "w") as file:
        file.write(ak.to_json(loaded))
        print(len(loaded))