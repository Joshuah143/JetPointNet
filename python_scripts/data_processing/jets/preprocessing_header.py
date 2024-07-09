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
    "rho": 'user.mswiatlo.40097513',
    "delta": 'user.mswiatlo.40097496',
    "JZ0": 'user.mswiatlo.39955613',
    "JZ1": 'user.mswiatlo.39955646',
    "JZ2": 'user.mswiatlo.39955678',
    "JZ3": 'user.mswiatlo.39955704',
    "JZ4": 'user.mswiatlo.39955735',
    "JZ5": 'user.mswiatlo.39955768',
    "JZ6": 'user.mswiatlo.39955825',
}
FILE_PREFIX_LEN = len(prefix_match['rho'])

prefix_to_set = {j:i for i,j in prefix_match.items()}


USER = Path.home().name
if USER == "luclissa":
    # ===== FIELDS TO CHANGE =====
    add_tracks_as_labels = False
    ENERGY_SCALE = 1
    MIN_TRACK_CELL_HITS = 25
    MAX_SAMPLE_LENGTH = 650
    DATA_PATH = ""

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
    NPZ_ALlOWED_SETS = ['JZ4', 'JZ5', 'JZ6']

    # ===== TRAINING =====
    TRAIN_DATASET_NAME = "attempt_1_june_18"
    TRAIN_OUTPUT_DIRECTORY_NAME = "full_set"
elif USER == "jhimmens":
    # ===== FIELDS TO CHANGE =====
    add_tracks_as_labels = False
    ENERGY_SCALE = 1
    MIN_TRACK_CELL_HITS = 25
    MAX_SAMPLE_LENGTH = 800
    DATA_PATH = Path("/fast_scratch_1/atlas/pflow/jhimmens_working_files")

    # ===== ROOT TO AWK =====
    AWK_OUTPUT_DIRECTORY_NAME = "rev_2"
    AWK_DATASET_NAME = "collected_data"
    OVERWRITE_AWK = False
    GEO_FILE_LOC = "/fast_scratch_1/atlas/pflow/rho_small.root"
    NUM_MAX_EVENTS_PER_CHUNK = 1000
    # TEST_SPLIT_RATIO is implied to be the remaining percentage
    TRAIN_SPLIT_RATIO = 0.55
    VAL_SPLIT_RATIO = 0.3
    AWK_THREADS_PER_CHUNK = 30  # root to awk
    ROOT_FILES_DIR = "/fast_scratch_1/atlas/pflow/20240614/user.mswiatlo.801166.Py8EG_A14NNPDF23LO_jj_JZ1.recon.ESD.e8514_e8528_s4185_s4114_r14977_2024.06.14.1_mltree.root" 

    # rho+delta: /fast_scratch_1/atlas/pflow/20240626/
    # dijet: /fast_scratch_1/atlas/pflow/20240614/
    # rho: /fast_scratch_1/atlas/pflow/20240626/user.mswiatlo.mc21_13p6TeV.900148.singlerho.recon.ESD.e8537_e8455_s3986_s3874_r14060_2024.06.26.v1_mltree.root
    # delta: /fast_scratch_1/atlas/pflow/20240626/user.mswiatlo.mc21_13p6TeV.900147.singleDelta.recon.ESD.e8537_e8455_s3986_s3874_r14060_2024.06.26.v1_mltree.root
    """
    user.mswiatlo.801165.Py8EG_A14NNPDF23LO_jj_JZ0.recon.ESD.e8514_e8528_s4185_s4114_r14977_2024.06.14.1_mltree.root
    user.mswiatlo.801166.Py8EG_A14NNPDF23LO_jj_JZ1.recon.ESD.e8514_e8528_s4185_s4114_r14977_2024.06.14.1_mltree.root
    user.mswiatlo.801167.Py8EG_A14NNPDF23LO_jj_JZ2.recon.ESD.e8514_e8528_s4185_s4114_r14977_2024.06.14.1_mltree.root
    user.mswiatlo.801168.Py8EG_A14NNPDF23LO_jj_JZ3.recon.ESD.e8514_e8528_s4185_s4114_r14977_2024.06.14.1_mltree.root
    user.mswiatlo.801169.Py8EG_A14NNPDF23LO_jj_JZ4.recon.ESD.e8514_e8528_s4185_s4114_r14977_2024.06.14.1_mltree.root
    user.mswiatlo.801170.Py8EG_A14NNPDF23LO_jj_JZ5.recon.ESD.e8514_e8528_s4185_s4114_r14977_2024.06.14.1_mltree.root
    user.mswiatlo.801171.Py8EG_A14NNPDF23LO_jj_JZ6.recon.ESD.e8514_e8528_s4185_s4114_r14977_2024.06.14.1_mltree.root
    """

    # ===== SAMPLE LENGTH SCRIPT =====
    LEN_OUTPUT_DIRECTORY_NAME = "rev_2"
    LEN_DATASET_NAME = "collected_data"
    SAMPLE_LENGTH_WORKERS = 20

    # ===== AWK TO NPZ =====
    NPZ_OUTPUT_DIRECTORY_NAME = "rev_2"
    NPZ_DATASET_NAME = "collected_data"
    OVERWRITE_NPZ = False
    NPZ_NUM_CHUNK_THREADS = 50  # awk to npz
    NPZ_ALlOWED_SETS = ['rho', 'delta', 'JZ2', 'JZ3', 'JZ4']

    # ===== TRAINING =====
    TRAIN_OUTPUT_DIRECTORY_NAME = "rev_2"
    TRAIN_DATASET_NAME = "collected_data"
    TRAIN_ALlOWED_SETS = ['JZ2', 'JZ3', 'JZ4'] #NOTE! This is just for the directory, we should add a filter to the data function?
else:
    raise Exception("User not found!")

NPZ_ALLOWED_PREFIXES = [prefix_match[i] for i in NPZ_ALlOWED_SETS]
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
TRAIN = 'train'
TUNE = 'tune'

def AWK_SAVE_LOC(working_file: str):
    if working_file == AWK:
        return (
            DATA_PATH
            / "pnet_data/processed_files"
            / AWK_DATASET_NAME
            / AWK_OUTPUT_DIRECTORY_NAME
            / "AwkwardArrs"
            / f"deltaR={MAX_DISTANCE}"
        )
    elif working_file == NPZ:
        return (
            DATA_PATH
            / "pnet_data/processed_files"
            / NPZ_DATASET_NAME
            / NPZ_OUTPUT_DIRECTORY_NAME
            / "AwkwardArrs"
            / f"deltaR={MAX_DISTANCE}"
        )
    elif working_file == LEN:
        return (
            DATA_PATH
            / "pnet_data/processed_files"
            / LEN_DATASET_NAME
            / LEN_OUTPUT_DIRECTORY_NAME
            / "AwkwardArrs"
            / f"deltaR={MAX_DISTANCE}"
        )
    else:
        raise Exception("File information not found")
        

def NPZ_SAVE_LOC(working_file: str):
    if working_file == NPZ:
        return (
            DATA_PATH
            / "pnet_data/processed_files"
            / NPZ_DATASET_NAME
            / NPZ_OUTPUT_DIRECTORY_NAME
            / "SavedNpz"
            / f"deltaR={MAX_DISTANCE}_maxLen={MAX_SAMPLE_LENGTH}_EScale={ENERGY_SCALE}"
        )
    elif working_file == TRAIN:
        return (
            DATA_PATH
            / "pnet_data/processed_files"
            / TRAIN_DATASET_NAME
            / TRAIN_OUTPUT_DIRECTORY_NAME
            / "SavedNpz"
            / f"deltaR={MAX_DISTANCE}_maxLen={MAX_SAMPLE_LENGTH}_EScale={ENERGY_SCALE}"
        )
    else:
        raise Exception("File information not found!")
