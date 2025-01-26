from multiprocessing import Pool
from pathlib import Path

import awkward as ak
import numpy as np
from loguru import logger as log
from load_from_root_file import load_from_root
from utils.data_loading import setup_directories, split_data
from utils.to_numpy import event_to_trainable


def save_train_data(*, config):
    input_data_dir = Path(config["data_pipeline"]["root_files_dir"])
    output_data_dir = Path(config["data_pipeline"]["output_dir"])
    geo_file = Path(config["global_params"]["geo_file_loc"])

    max_delta_r = config["data_pipeline"]["max_delta_r"]
    desired_sets = config["data_pipeline"]["sets_to_process"]
    set_to_dir_name = config["data_pipeline"]["set_paths"]
    data_split = config["data_pipeline"]["splits"]
    save_location = Path(config["data_pipeline"]["output_dir"])
    chunk_size = config["data_pipeline"]["augmented"]["chunk_size"]
    max_event_len = config["global_params"]["max_sample_length"]

    log.info("Processing augmented data pipeline")
    setup_directories(
        save_location,
        desired_sets,
        data_split,
        run_id=config["global_params"]["run_id"],
    )
    save_location = save_location / config["global_params"]["run_id"]

    for set_name in desired_sets:
        log.info(f"Handling set: {set_name}")
        log.info(f"Loading data from: {input_data_dir/set_to_dir_name[set_name]}")

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
                tasks.append(
                    (
                        split_type_name,
                        chunk,
                        split_save_location,
                        start_idx,
                        max_delta_r,
                        max_event_len,
                    )
                )
        with Pool() as pool:
            pool.map(_process_split, tasks)


def _process_split(args: tuple[str, ak.Array, Path, int, float, int]):
    (split_type, data, split_save_location, split_id, max_delta_r, max_event_len) = args

    iters = 0
    while max(ak.num(data["tracks"])) > 0:  # while there are still tracks in the data
        log.debug(
            f"Processing split {split_id} iteration {iters}, {len(data)} events remaining"
        )
        trainable = ak_to_numpy(data, max_delta_r, max_event_len)
        np.save(split_save_location / f"{iters}_reductions__{split_id}", trainable)
        # this should be able to be saved as parquet instead of json, but it runs into an issue inside ak
        ak.to_json(
            data,
            split_save_location / f"{iters}_reductions_{split_id}.json",
            line_delimited=True,
        )
        iters += 1
        data = perform_subtraction(data)
        # current: old events are not removed from the data, all saved files have the same size, this is a large issue


def ak_to_numpy(ak_array: ak.Array, max_delta_r: float, max_event_len: int):
    # for each event run the event_to_trainable function
    trainable = []
    for event in ak_array:
        if len(event["tracks"]) == 0:
            continue  # remove events with on tracks
        focal_index = ak.argmax(event["tracks"]["trackPt"])  # selected by pT
        trainable.append(
            event_to_trainable(
                event,
                focal_index=focal_index,
                delta_r_max=max_delta_r,
                max_event_len=max_event_len,
            )
        )
    return np.array(trainable)


def perform_subtraction(ak_array: ak.Array):
    b = ak.ArrayBuilder()
    for event in ak_array:
        single_event_subtraction(b, event)
    return b.snapshot()


# This could be more efficient by, but the mutability of the Record becomes an issue
def single_event_subtraction(b: ak.ArrayBuilder, event: ak.Record):
    if len(event["tracks"]) == 0:
        log.debug(f'event {event["eventNumber"]} skipped due to no remaining tracks')
        log.debug(f'{len(event["attributed"])} tracks attributed')
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
