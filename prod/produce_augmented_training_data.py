from multiprocessing import Pool
from pathlib import Path
import awkward as ak
import numpy as np

from load_from_root_file import load_from_root
from to_numpy import event_to_trainable

# Meta params
input_data_dir = Path(
    "/Users/jhimmens/Library/CloudStorage/Dropbox/Work/TRIUMF/jetpointnet/prod/data"
)  # directory for data to be taken from
output_data_dir = Path(
    "/Users/jhimmens/Library/CloudStorage/Dropbox/Work/TRIUMF/jetpointnet/prod/training_data"
)  # path for saved data to be written to
geo_file = Path(
    "/Users/jhimmens/Library/CloudStorage/Dropbox/Work/TRIUMF/jetpointnet/prod/data/rho_small.root"
)  # a file with a cell_geo_tree

# input_data_dir = Path("/fast_scratch_3/atlas/pflow/ntuples/20240916.v0/") # directory for data to be taken from
# output_data_dir = Path("/fast_scratch_3/atlas/pflow/augmented_training_data") # path for saved data to be written to
# geo_file = Path("/fast_scratch_1/atlas/pflow/rho_small.root") # a file with a cell_geo_tree


max_sample_length = 800
desired_sets = [
    # "rho",
    # "delta",
    # "JZ0",
    # "JZ1",
    # "JZ2",
    # "JZ3",
    "JZ4",
    #     "JZ5",
    #     "JZ6",
    #     "JZ7",
    #     "JZ8",
    #     "JZ9",
]
set_to_dir_name = {
    #     "JZ0": "user.jhimmens.801165.Py8EG_A14NNPDF23LO_jj_JZ0.recon.ESD.e8514_e8528_s4185_s4114_r14977_20240916.v0_mltree.root",
    #     "JZ1": "user.jhimmens.801166.Py8EG_A14NNPDF23LO_jj_JZ1.recon.ESD.e8514_e8528_s4185_s4114_r14977_20240916.v0_mltree.root",
    #     "JZ2": "user.jhimmens.801167.Py8EG_A14NNPDF23LO_jj_JZ2.recon.ESD.e8514_e8528_s4185_s4114_r14977_20240916.v0_mltree.root",
    #     "JZ3": "user.jhimmens.801168.Py8EG_A14NNPDF23LO_jj_JZ3.recon.ESD.e8514_e8528_s4185_s4114_r14977_20240916.v0_mltree.root",
    "JZ4": "JZ4",
    #     "JZ4": "user.jhimmens.801169.Py8EG_A14NNPDF23LO_jj_JZ4.recon.ESD.e8514_e8528_s4185_s4114_r14977_20240916.v0_mltree.root",
    #     "JZ5": "user.jhimmens.801170.Py8EG_A14NNPDF23LO_jj_JZ5.recon.ESD.e8514_e8528_s4185_s4114_r14977_20240916.v0_mltree.root",
    #     "JZ6": "user.jhimmens.801171.Py8EG_A14NNPDF23LO_jj_JZ6.recon.ESD.e8514_e8528_s4185_s4114_r14977_20240916.v0_mltree.root",
    #     "JZ7": "user.jhimmens.801172.Py8EG_A14NNPDF23LO_jj_JZ7.recon.ESD.e8514_e8528_s4185_s4114_r14977_20240916.v0_mltree.root",
    #     "JZ8": "user.jhimmens.801173.Py8EG_A14NNPDF23LO_jj_JZ8.recon.ESD.e8514_e8528_s4185_s4114_r14977_20240916.v0_mltree.root",
    #     "JZ9": "user.jhimmens.801174.Py8EG_A14NNPDF23LO_jj_JZ9incl.recon.ESD.e8514_e8528_s4185_s4114_r14977_20240916.v0_mltree.root",
    #     "delta": "user.jhimmens.mc21_13p6TeV.900147.singleDelta.recon.ESD.e8537_e8455_s3986_s3874_r14060_20240916.v0_mltree.root",
    #     "rho": "user.jhimmens.mc21_13p6TeV.900148.singlerho.recon.ESD.e8537_e8455_s3986_s3874_r14060_20240916.v0_mltree.root",
}
data_split = {"train": 0.6, "val": 0.2, "test": 0.2}


def save_train_data(save_location: Path, chunk_size: int = 100):
    setup_directories(save_location)
    for set_name in desired_sets:
        print(f"Handling set: {set_name}")
        print(f"Loading data from: {input_data_dir/set_to_dir_name[set_name]}")

        root_tree: ak.Array = load_from_root(
            input_data_dir / set_to_dir_name[set_name] / "*.root", geo_file, debug=False
        )

        split_tree = split_data(root_tree, data_split)
        tasks = []
        for split_type_name, data in split_tree.items():  # ('test', uproot items)
            # Can iterate over the data and save it in chunks, data is an awkward array of records
            iters = 0
            split_save_location = save_location / split_type_name / set_name

            for start_idx in range(0, len(data), chunk_size):
                chunk = data[start_idx : start_idx + chunk_size]
                split_save_location = save_location / split_type_name / set_name
                tasks.append((split_type_name, chunk, split_save_location, start_idx))
        with Pool() as pool:
            pool.map(process_split, tasks)


def process_split(args):
    split_type, data, split_save_location, split_id = args
    iters = 0
    while max(ak.num(data["tracks"])) > 0:  # while there are still tracks in the data
        print(f"{len(data)} event remaining after {iters} iterations")
        trainable = ak_to_numpy(data)
        np.save(split_save_location / f"{iters}_reductions__{split_id}.npz", trainable)
        # this should be able to be saved as parquet instead of json, but it runs into an issue inside ak
        ak.to_json(
            data,
            split_save_location / f"{iters}_reductions_{split_id}.json",
            line_delimited=True,
        )
        iters += 1
        data = perform_subtraction(data)
        # current: old events are not removed from the data, all saved files have the same size, this is a large issue


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
        if len(event["tracks"]) == 0:
            continue  # remove events with on tracks
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
    if len(event["tracks"]) == 0:
        # print(f'event {event["eventNumber"]} skipped due to no remaining tracks')
        # print(f'{len(event["attributed"])} tracks attributed')
        return
    b.begin_record()

    max_pt_index = ak.argmax(event["tracks"]["trackPt"])
    truth_part_idx = event["tracks"][max_pt_index]["trackTruthParticleIndex"]

    b.field("runNumber")
    b.integer(event["runNumber"])
    b.field("eventNumber")
    b.integer(event["eventNumber"])
    b.field("lumiBlock")
    b.integer(event["lumiBlock"])
    b.field("coreFlags")

    b.field("tracks")
    b.begin_list()
    for i, track in enumerate(event["tracks"]):
        if i != max_pt_index:
            b.append(track)
    b.end_list()

    removed_cells = []

    b.field("cells")
    b.begin_list()
    for i, cell in enumerate(event["cells"]):
        if truth_part_idx in cell["cell_hitsTruthIndex"]:
            index_in_cell_truth_focal = ak.where(
                cell["cell_hitsTruthIndex"] == truth_part_idx
            )[0][0]

            focal_truth_e = cell["cell_hitsTruthE"][index_in_cell_truth_focal]
            ratio = focal_truth_e / cell["cell_E"]
            updated_E = cell["cell_E"] * (1 - ratio)
            removed_E = cell["cell_E"] * ratio

            removed_cells.append(
                {
                    "ID": cell["cell_ID"],
                    "E": removed_E,
                    "x": cell["x"],
                    "y": cell["y"],
                    "z": cell["z"],
                    "eta": cell["eta"],
                    "phi": cell["phi"],
                    "sigma": cell["cell_sigma"],
                }
            )

            b.begin_record()
            b.field("cell_cluster_index")
            b.integer(cell["cell_cluster_index"])
            b.field("cell_E")
            b.real(updated_E)
            b.field("cell_ID")
            b.integer(cell["cell_ID"])
            b.field("cell_sigma")
            b.real(cell["cell_sigma"])
            b.field("eta")
            b.real(cell["eta"])
            b.field("phi")
            b.real(cell["phi"])
            b.field("x")
            b.real(cell["x"])
            b.field("y")
            b.real(cell["y"])
            b.field("z")
            b.real(cell["z"])
            b.field("cell_hitsTruthIndex")
            b.begin_list()
            for idx in cell["cell_hitsTruthIndex"]:
                if idx != truth_part_idx:
                    b.integer(idx)
            b.end_list()
            b.field("cell_hitsTruthE")
            b.begin_list()
            for j in range(len(cell["cell_hitsTruthE"])):
                if j != index_in_cell_truth_focal:
                    b.real(cell["cell_hitsTruthE"][j])
            b.end_list()
            b.field("cell_hitsTruthTotalE")
            b.real(cell["cell_hitsTruthTotalE"] - focal_truth_e)
            b.end_record()
        else:
            b.append(cell)
    b.end_list()
    b.field("attributed")
    b.begin_list()
    b.begin_record()
    b.field("track")
    # FIX BELOW
    b.append(event["tracks"][max_pt_index])
    b.field("cells")
    b.begin_list()
    for cell in removed_cells:
        b.begin_record()
        b.field("ID")
        b.integer(cell["ID"])
        b.field("E")
        b.real(cell["E"])
        b.field("x")
        b.real(cell["x"])
        b.field("y")
        b.real(cell["y"])
        b.field("z")
        b.real(cell["z"])
        b.field("eta")
        b.real(cell["eta"])
        b.field("phi")
        b.real(cell["phi"])
        b.field("sigma")
        b.real(cell["sigma"])
        b.end_record()

    b.end_list()
    b.end_record()
    b.end_list()
    b.end_record()


if __name__ == "__main__":
    save_train_data(output_data_dir)
