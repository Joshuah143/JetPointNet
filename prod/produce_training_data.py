from pathlib import Path
import awkward as ak
import numpy as np
from keras.src.legacy.backend import update
from matplotlib.style.core import update_nested_dict

from prod.load_from_root_file import load_from_root
from prod.to_numpy import event_to_trainable

# Meta params
input_data_dir = Path("/Users/joshuahimmens/Library/CloudStorage/Dropbox/Work/TRIUMF/jetpointnet/prod/data")
output_data_dir = Path("/Users/joshuahimmens/Library/CloudStorage/Dropbox/Work/TRIUMF/jetpointnet/prod/training_data")
max_sample_length = 800
desired_sets = [
    "rho",
    # "delta",
    # "JZ0",
    # "JZ1",
    # "JZ2",
    # "JZ3",
    # "JZ4",
    # "JZ5",
    # "JZ6",
    # "JZ7",
]
data_split = {
    'train': 0.6,
    'val': 0.2,
    'test': 0.2
}

def save_train_data(save_location: Path):
    setup_directories(save_location)
    for set_name in desired_sets:
        print(f"Handling set: {set_name}")
        print(f"Loading data from: {input_data_dir/set_name}")
        tree: ak.Array = load_from_root(input_data_dir/set_name/"*.root")

        # sort into data_split sets
        split_tree = split_data(tree, data_split)
        for split_type, data in split_tree.items():
            split_save_location = save_location / split_type / set_name
            split_save_location.mkdir(exist_ok=True)
            iters = 0
            while max(ak.num(data['tracks'])) > 0: # while there are still tracks in the data
                trainable = ak_to_numpy(data)
                np.save(split_save_location / f"{iters}_reductions.npy", trainable)
                iters += 1
                data = perform_subtraction(data)



def setup_directories(save_location: Path):
    save_location.mkdir(exist_ok=True)
    for split_type in data_split.keys():
        split_save_location = save_location / split_type
        split_save_location.mkdir(exist_ok=True)
        for set_name in desired_sets:
            set_save_location = split_save_location / set_name
            set_save_location.mkdir(exist_ok=True)


def split_data(tree: ak.Array, data_splits: dict):
    split_tree = {}
    current_idx = 0
    end_idx = len(tree)
    for split_type, split_ratio in data_splits.items():
        subset_term = int(split_ratio * end_idx) + current_idx
        split_tree[split_type] = tree[current_idx:subset_term]
        current_idx = subset_term
    return split_tree

def ak_to_numpy(ak_array: ak.Array):
    # for each event run the event_to_trainable function
    trainable = []
    for event in ak_array:
        if len(event['tracks']) == 0:
            continue # remove events with on tracks
        trainable.append(event_to_trainable(event))
    return np.array(trainable)

def perform_subtraction(ak_array: ak.Array):
    # remove the first max_sample_length elements from the array
    b = ak.ArrayBuilder()
    for event in ak_array:
        single_event_subtraction(b, event)
    return b.snapshot()


# This could be more efficient by, but the mutability of the Record becomes an issue
def single_event_subtraction(b: ak.ArrayBuilder, event: ak.Record):
    b.begin_record()

    max_pt_index = ak.argmax(event['tracks']['trackPt'])
    truth_part_idx = event['tracks'][max_pt_index]['trackTruthParticleIndex']

    b.field("runNumber")
    b.integer(event['runNumber'])
    b.field("eventNumber")
    b.integer(event['eventNumber'])
    b.field("lumiBlock")
    b.integer(event['lumiBlock'])
    b.field("coreFlags")

    b.field("tracks")
    b.begin_list()
    for i, track in enumerate(event['tracks']):
        if i != max_pt_index:
            b.append(track)
    b.end_list()

    removed_cells = []

    b.field("cells")
    b.begin_list()
    for i, cell in enumerate(event['cells']):
        if truth_part_idx in cell['cell_hitsTruthIndex']:
            index_in_cell_truth_focal = ak.where(cell['cell_hitsTruthIndex'] == truth_part_idx)[0][0]

            focal_truth_e = cell['cell_hitsTruthE'][index_in_cell_truth_focal]
            ratio = focal_truth_e / cell['cell_E']
            updated_E = cell['cell_E'] * (1 - ratio)
            removed_E = cell['cell_E'] * ratio

            removed_cells.append({"ID": cell['cell_ID'],
                                  "E": removed_E,
                                 'x': cell['x'],
                                  'y': cell['y'],
                                  'z': cell['z'],
                                  'eta': cell['eta'],
                                  'phi': cell['phi'],
                                  'sigma': cell['cell_sigma']})

            # TODO: add normalized params
            b.begin_record()
            b.field("cell_cluster_index")
            b.integer(cell['cell_cluster_index'])
            b.field("cell_E")
            b.real(updated_E)
            b.field("cell_ID")
            b.integer(cell['cell_ID'])
            b.field("cell_sigma")
            b.real(cell['cell_sigma'])
            b.field("eta")
            b.real(cell['eta'])
            b.field("phi")
            b.real(cell['phi'])
            b.field("x")
            b.real(cell['x'])
            b.field("y")
            b.real(cell['y'])
            b.field("z")
            b.real(cell['z'])
            b.field("cell_hitsTruthIndex")
            b.begin_list()
            for idx in cell['cell_hitsTruthIndex']:
                if idx != truth_part_idx:
                    b.integer(idx)
            b.end_list()
            b.field("cell_hitsTruthE")
            b.begin_list()
            for j in range(len(cell['cell_hitsTruthE'])):
                if j != index_in_cell_truth_focal:
                    b.real(cell['cell_hitsTruthE'][j])
            b.end_list()
            b.field("cell_hitsTruthTotalE")
            b.real(cell['cell_hitsTruthTotalE'] - focal_truth_e)
            b.end_record()
        else:
            b.append(cell)
    b.end_list()
    b.field("attributed")
    b.begin_list()
    b.begin_record()
    b.field('track')
    b.append(event['tracks'][max_pt_index])
    b.field('cells')
    b.begin_list()
    for cell in removed_cells:
        b.begin_record()
        b.field('ID')
        b.integer(cell['ID'])
        b.field('E')
        b.real(cell['E'])
        b.field('x')
        b.real(cell['x'])
        b.field('y')
        b.real(cell['y'])
        b.field('z')
        b.real(cell['z'])
        b.field('eta')
        b.real(cell['eta'])
        b.field('phi')
        b.real(cell['phi'])
        b.field('sigma')
        b.real(cell['sigma'])
        b.end_record()

    b.end_list()
    b.end_record()
    b.end_list()
    b.end_record()

