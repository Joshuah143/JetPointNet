import sys
from pathlib import Path
import os

# CERNBOX = os.environ["CERNBOX"]
REPO_PATH = Path.home() / "workspace/jetpointnet"
SCRIPT_PATH = REPO_PATH / "python_scripts"
sys.path.append(str(SCRIPT_PATH))

from data_processing.jets.track_metadata import (
    calo_layers,
    has_fixed_r,
    fixed_r,
    fixed_z,
)

HAS_FIXED_R, FIXED_R, FIXED_Z = (
    has_fixed_r,
    fixed_r,
    fixed_z,
)  # Loading the calorimeter geometery

prefix_match = {
    "JZ0": 'user.mswiatlo.39955613',
    "JZ1": 'user.mswiatlo.39955646',
    "JZ2": 'user.mswiatlo.39955678',
    "JZ3": 'user.mswiatlo.39955704',
    "JZ4": 'user.mswiatlo.39955735',
    "JZ5": 'user.mswiatlo.39955768',
    "JZ6": 'user.mswiatlo.39955825',
}

prefix_to_set = {j:i for i, j in prefix_match.items()}


USER = Path.home().name
if USER == "luclissa":
    # ===== FIELDS TO CHANGE =====
    add_tracks_as_labels = False
    ENERGY_SCALE = 1

    # ===== ROOT TO AWK =====
    AWK_OUTPUT_DIRECTORY_NAME = "500k_events_june19/"
    AWK_DATASET_NAME = "cern_grid"
    OVERWRITE_AWK = False
    GEO_FILE_LOC = "/data/atlas/data/rho_delta/rho_small.root"
    NUM_MAX_EVENTS_PER_CHUNK = 1000
    # TEST_SPLIT_RATIO is implied to be the remaining percentage
    TRAIN_SPLIT_RATIO = 0.55
    VAL_SPLIT_RATIO = 0.3
    AWK_THREADS_PER_CHUNK = 25  # root to awk
    ROOT_FILES_DIR ="/eos/home-m/mswiatlo/forLuca/mltree_large.root" # NOTE: You will need to change this to a directory containing this file to be compatible

    # ===== SAMPLE LENGTH SCRIPT =====
    LEN_DATASET_NAME = "cern_grid"
    LEN_OUTPUT_DIRECTORY_NAME = "500k_events_june19/"
    SAMPLE_LENGTH_WORKERS = 5
    DATASET_NAME = "cern_grid"

    # ===== AWK TO NPZ =====
    NPZ_DATASET_NAME = "cern_grid"
    NPZ_OUTPUT_DIRECTORY_NAME = "500k_events_june19/"
    OVERWRITE_NPZ = False
    NPZ_NUM_CHUNK_THREADS = 25  # awk to npz
    DATASET_NAME = "cern_grid"
    MAX_SAMPLE_LENGTH = 650
    NPZ_REGEX_INCLUDE = r".*" # all awk files included
elif USER == "jhimmens":
    # ===== FIELDS TO CHANGE =====
    add_tracks_as_labels = False
    ENERGY_SCALE = 1

    # ===== ROOT TO AWK =====
    AWK_OUTPUT_DIRECTORY_NAME = "rho_small/"
    AWK_DATASET_NAME = "progressive_training"
    OVERWRITE_AWK = False
    GEO_FILE_LOC = "/fast_scratch_1/atlas/pflow/rho_small.root"
    NUM_MAX_EVENTS_PER_CHUNK = 200
    # TEST_SPLIT_RATIO is implied to be the remaining percentage
    TRAIN_SPLIT_RATIO = 0.55
    VAL_SPLIT_RATIO = 0.3
    AWK_THREADS_PER_CHUNK = 20  # root to awk
    ROOT_FILES_DIR = "/fast_scratch_1/atlas/pflow/delta.root" # "/fast_scratch_1/atlas/pflow/20240614/" full set, rho delta

    # ===== SAMPLE LENGTH SCRIPT =====
    LEN_DATASET_NAME = "cern_grid"
    LEN_OUTPUT_DIRECTORY_NAME = "500k_events_june19/"
    SAMPLE_LENGTH_WORKERS = 10

    # ===== AWK TO NPZ =====
    NPZ_DATASET_NAME = "cern_grid"
    NPZ_OUTPUT_DIRECTORY_NAME = "500k_events_june19/"
    OVERWRITE_NPZ = False
    NPZ_NUM_CHUNK_THREADS = 1  # awk to npz
    MAX_SAMPLE_LENGTH = 650
    NPZ_REGEX_INCLUDE = f'^({prefix_match["JZ0"]}|{prefix_match["JZ1"]}|{prefix_match["JZ2"]}).*' # all awk files included
else:
    raise Exception("User not found!")


POINT_TYPE_LABELS = {0: "focus hit", 1: "cell", 2: "unfocus hit", -1: "padding"}
POINT_TYPE_ENCODING = {v: k for k, v in POINT_TYPE_LABELS.items()}

DEBUG_NUM_EVENTS_TO_USE = None
UPROOT_MASK_VALUE_THRESHOLD = -100_000
MAX_DISTANCE = 0.2  # could potentially go up to about 0.5 as a hard max

# Path to the ROOT file containing jet events

# RHO DATA
# FILE_LOC = "/eos/home-m/mswiatlo/images/truthPerCell/rho_full.root"
# GEO_FILE_LOC = "/eos/home-m/mswiatlo/images/truthPerCell/cell_geo.root"

# JETS DATA
# FILE_LOC = "/data/atlas/mltree_2000_fixedHits.root"
# GEO_FILE_LOC = "/data/atlas/data/rho_delta/rho_small.root"
# FILE_LOC = "/eos/home-m/mswiatlo/forLuca/mltree_large.root"
# GEO_FILE_LOC = "/eos/home-m/mswiatlo/images/truthPerCell/cell_geo.root"

AWK = 'awk'
NPZ = 'npz'
LEN = 'len'

def AWK_SAVE_LOC(working_file: str):
    if working_file == AWK:
        return (
            REPO_PATH
            / "pnet_data/processed_files"
            / AWK_DATASET_NAME
            / AWK_OUTPUT_DIRECTORY_NAME
            / "AwkwardArrs"
            / f"deltaR={MAX_DISTANCE}"
        )
    elif working_file == NPZ:
        return (
            REPO_PATH
            / "pnet_data/processed_files"
            / NPZ_DATASET_NAME
            / NPZ_OUTPUT_DIRECTORY_NAME
            / "AwkwardArrs"
            / f"deltaR={MAX_DISTANCE}"
        )
    elif working_file == LEN:
        return (
            REPO_PATH
            / "pnet_data/processed_files"
            / LEN_DATASET_NAME
            / LEN_OUTPUT_DIRECTORY_NAME
            / "AwkwardArrs"
            / f"deltaR={MAX_DISTANCE}"
        )
    else:
        raise Exception("File information not found")
        

def NPZ_SAVE_LOC(working_file: str):
    if working_file == AWK:
        return (
            REPO_PATH
            / "pnet_data/processed_files"
            / AWK_DATASET_NAME
            / AWK_OUTPUT_DIRECTORY_NAME
            / "SavedNpz"
            / f"deltaR={MAX_DISTANCE}_maxLen={MAX_SAMPLE_LENGTH}"
            / f"{ENERGY_SCALE=}".lower()
        )
    elif working_file == NPZ:
        return (
            REPO_PATH
            / "pnet_data/processed_files"
            / NPZ_DATASET_NAME
            / NPZ_OUTPUT_DIRECTORY_NAME
            / "SavedNpz"
            / f"deltaR={MAX_DISTANCE}_maxLen={MAX_SAMPLE_LENGTH}"
            / f"{ENERGY_SCALE=}".lower()
        )
    elif working_file == LEN:
        return (
            REPO_PATH
            / "pnet_data/processed_files"
            / LEN_DATASET_NAME
            / LEN_OUTPUT_DIRECTORY_NAME
            / "SavedNpz"
            / f"deltaR={MAX_DISTANCE}_maxLen={MAX_SAMPLE_LENGTH}"
            / f"{ENERGY_SCALE=}".lower()
        )
    else:
        raise Exception("File information not found!")
        