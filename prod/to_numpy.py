import awkward as ak
import numpy as np
from coordinate_conversions import calculate_delta_r

SENTINEL_NO_DATA = -1
POINT_TYPE_LABELS = {0: "focal_track", 1: "cell", 2: "non_focal_track", SENTINEL_NO_DATA: "padding"}
POINT_TYPE_ENCODING = {v: k for k, v in POINT_TYPE_LABELS.items()}


event_array_dtype = np.dtype([
    ('runNumber', np.int32),
    ('eventNumber', np.int32),
    ('cell_ID', np.int32),
    ('track_ID', np.int32),

    # SENTINEL_NO_DATA if non-truth event
    ('truth_cell_focal_fraction_energy', np.float32),
    ('truth_cell_non_focal_fraction_energy', np.float32),
    ('truth_cell_neutral_fraction_energy', np.float32),
    ('truth_cell_total_energy', np.float32),

    ('category', np.int8),
    ('track_num', np.int32),
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('delta_R', np.float32),
    ('eta', np.float32),
    ('phi', np.float32),

    # ('distance', np.float32), was included... is it still required? Hard to calculate
    ('normalized_x', np.float32),
    ('normalized_y', np.float32),
    ('normalized_z', np.float32),

    ('cell_sigma', np.float32),
    ('track_chi2_dof', np.float32),
    # ("track_chi2_dof_cell_sigma", np.float32), TBD if combined classes are required/useful
    ('cell_E', np.float32),
    ('normalized_cell_E', np.float32),
    ('track_pt', np.float32),
    ('normalized_track_pt', np.float32),
    # ('track_pt_cell_E', np.float32), TBD if combined classes are required/useful
    # ('normalized_track_pt_cell_E', np.float32), TBD if combined classes are required/useful
])

def event_to_trainable(event_ak: ak.Array, delta_R_max=0.2, truth=True, max_event_len=800) -> np.ndarray:
    trainable_array = []
    for event in event_ak:
        # handle focal track
        print(event.show(type=True))
        focal_index = ak.argmax(event['tracks']['trackPt']) # selected by pT
        focal_track = event['tracks'][focal_index]
        focal_eta = focal_track['trackEta']
        focal_phi = focal_track['trackPhi']

        # handle other tracks
        all_track_eta = event['tracks']['trackEta']
        all_track_phi = event['tracks']['trackPhi']
        delta_R = calculate_delta_r(eta1=focal_eta, phi1=focal_phi, eta2=all_track_eta, phi2=all_track_phi)
        track_indices = ak.local_index(all_track_eta)
        within_deltaR = (delta_R < delta_R_max) & (track_indices != focal_index)
        adjacent_tracks = event['tracks'][within_deltaR]

        # Handle cells within delta R
        cell_eta = event['cells']['eta']
        cell_phi = event['cells']['phi']
        delta_R_cells = calculate_delta_r(eta1=cell_eta, phi1=cell_phi, eta2=focal_eta, phi2=focal_phi)
        mask_cells = delta_R_cells < delta_R_max
        adjacent_cells = event['cells'][mask_cells]

        # run processing
        focal_data = add_focal_track(focal_track, truth=truth)
        track_data = add_non_focal_tracks(adjacent_tracks, focal_track, truth=truth)
        # TODO: sort by max E so that cutoff does not cause issues
        cell_data = add_cells(adjacent_cells, focal_track, truth=truth)

        # combine fields
        event_data = np.concatenate([focal_data, track_data, cell_data])
        event_data['runNumber'] = event['runNumber']
        event_data['eventNumber'] = event['eventNumber']

        event_data = truncate_or_pad(event_data, max_event_len)
        # TODO: normalize the data

        trainable_array.append(event_data)

        # Convert to numpy array
    trainable_array = np.stack(trainable_array)
    return trainable_array

def normalize_event_data(event_data):
    # Normalize x, y, z
    positions = np.vstack((event_data['x'], event_data['y'], event_data['z'])).T
    norms = np.linalg.norm(positions, axis=1)
    norms[norms == 0] = 1  # Avoid division by zero

    event_data['normalized_x'] = event_data['x'] / norms
    event_data['normalized_y'] = event_data['y'] / norms
    event_data['normalized_z'] = event_data['z'] / norms

    # Normalize cell_E and track_pt
    # For cells
    cell_E_valid = event_data['cell_E'][event_data['cell_E'] != -1]
    if len(cell_E_valid) > 0:
        max_cell_E = np.max(cell_E_valid)
    else:
        max_cell_E = 1  # Avoid division by zero
    event_data['normalized_cell_E'] = np.where(event_data['cell_E'] != -1, event_data['cell_E'] / max_cell_E, -1)

    # For tracks
    track_pt_valid = event_data['track_pt'][event_data['track_pt'] != -1]
    if len(track_pt_valid) > 0:
        max_track_pt = np.max(track_pt_valid)
    else:
        max_track_pt = 1  # Avoid division by zero
    event_data['normalized_track_pt'] = np.where(event_data['track_pt'] != -1, event_data['track_pt'] / max_track_pt, -1)

    return event_data


def truncate_or_pad(event_data, max_event_len):
    current_len = len(event_data)
    if current_len > max_event_len: # truncate
        event_data = event_data[:max_event_len]
    elif current_len < max_event_len: # pad
        pad_length = max_event_len - current_len
        padding = np.zeros(pad_length, dtype=event_array_dtype)
        padding.fill(SENTINEL_NO_DATA)
        event_data = np.concatenate([event_data, padding])
    return event_data

def add_focal_track(track, truth):
    return np.zeros(0, dtype=event_array_dtype)

def add_non_focal_tracks(tracks, focal_track, truth):
    hits_per_track = ak.num(tracks['hits'])
    total_hits = ak.sum(hits_per_track)

    trainable = np.zeros(total_hits, dtype=event_array_dtype)
    trainable.fill(SENTINEL_NO_DATA)

    trainable['category'] = POINT_TYPE_ENCODING['non_focal_track']

    # track data
    chi2 = ak.to_numpy(tracks['trackChiSquared'])
    ndof = ak.to_numpy(tracks['trackNumberDOF'])
    track_pt = ak.to_numpy(tracks['trackPt'])
    track_IDs = np.arange(1, len(tracks) + 1)

    # repeat for array fitting
    chi2_by_ndof_per_hit = np.repeat(chi2/ndof, hits_per_track)
    track_pt_per_hit = np.repeat(track_pt, hits_per_track)
    track_IDs_per_hit = np.repeat(track_IDs, hits_per_track)

    # hit data
    x = ak.to_numpy(ak.flatten(tracks['hits']['x']))
    y = ak.to_numpy(ak.flatten(tracks['hits']['y']))
    z = ak.to_numpy(ak.flatten(tracks['hits']['z']))
    eta = ak.to_numpy(ak.flatten(tracks['hits']['eta']))
    phi = ak.to_numpy(ak.flatten(tracks['hits']['phi']))

    trainable['x'] = x
    trainable['y'] = y
    trainable['z'] = z
    trainable['eta'] = eta
    trainable['phi'] = phi
    trainable['track_ID'] = track_IDs_per_hit
    trainable['track_chi2_dof'] = chi2_by_ndof_per_hit
    trainable['track_pt'] = track_pt_per_hit

    delta_R = calculate_delta_r(eta, focal_track['trackEta'], phi, focal_track['trackPhi'])
    trainable['delta_R'] = delta_R

    return trainable

def add_cells(cells, focal_track, truth: bool):
    cell_data = np.zeros(0, dtype=event_array_dtype)
    cell_data.fill(SENTINEL_NO_DATA)
    return cell_data

    cell_data['category'] = POINT_TYPE_ENCODING['cell']

    # Extract arrays
    x = ak.to_numpy(cells['x'])
    y = ak.to_numpy(cells['y'])
    z = ak.to_numpy(cells['z'])
    eta = ak.to_numpy(cells['eta'])
    phi = ak.to_numpy(cells['phi'])
    cell_sigma = ak.to_numpy(cells['cell_sigma'])
    cell_E = ak.to_numpy(cells['cell_E'])
    cell_ID = ak.to_numpy(cells['cell_ID'])

    cell_data['cell_ID'] = cell_ID
    cell_data['x'] = x
    cell_data['y'] = y
    cell_data['z'] = z
    cell_data['eta'] = eta
    cell_data['phi'] = phi
    cell_data['cell_sigma'] = cell_sigma
    cell_data['cell_E'] = cell_E

    # Calculate delta_R between cells and focal track
    # delta_eta = eta - focal_track['trackEta']
    # delta_phi = phi - focal_track['trackPhi']
    # delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    # delta_R = np.sqrt(delta_eta ** 2 + delta_phi ** 2)
    delta_R = calculate_delta_r(eta, focal_track['trackEta'], phi,focal_track['trackPhi'])
    cell_data['delta_R'] = delta_R

    # Other fields can be set to default or zero
    return cell_data


    # Choose highest pT track

    # create array

    # add highest pT track

    # add tracks within delta R

    # normalize E, pT, x, y, z
