import sys
from pathlib import Path

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


# ===== FIELDS TO CHANGE =====
USER = Path.home().name
if USER == "jhimmens":
    add_tracks_as_labels = False
    NUM_EVENTS_PER_CHUNK = 200
    TRAIN_SPLIT_RATIO = 0.55
    VAL_SPLIT_RATIO = 0.3
    # TEST_SPLIT_RATIO is implied to be the remaining percentage
    NUM_THREAD_PER_CHUNK = 25  # root to awk
    NUM_CHUNK_THREADS = 30  # awk to npz
    # OUTPUT_DIRECTORY_NAME = "rho_full/"
    # OUTPUT_DIRECTORY_NAME = "ttbar"
    OUTPUT_DIRECTORY_NAME = "2000_events_w_fixed_hits/"
    # DATASET_NAME = "benchmark"
    DATASET_NAME = "large_R"
    FILE_LOC = "/fast_scratch_1/atlas/pflow/mltree_2000_fixedHits.root"
    GEO_FILE_LOC = "/fast_scratch_1/atlas/pflow/rho_small.root"
    ENERGY_SCALE = 1000
elif USER == "luclissa":
    add_tracks_as_labels = False
    NUM_EVENTS_PER_CHUNK = 200
    TRAIN_SPLIT_RATIO = 0.55
    VAL_SPLIT_RATIO = 0.3
    # TEST_SPLIT_RATIO is implied to be the remaining percentage
    NUM_THREAD_PER_CHUNK = 25  # root to awk
    NUM_CHUNK_THREADS = 25  # awk to npz
    DATASET_NAME = "ttbar"  # or "rho_full/"
    OUTPUT_DIRECTORY_NAME = "benchmark"  # or "raw": NOTE: for tests change this
    FILE_LOC = "/eos/home-m/mswiatlo/forLuca/mltree_large.root"
    GEO_FILE_LOC = "/eos/home-m/mswiatlo/images/truthPerCell/cell_geo.root"
    ENERGY_SCALE = 1000
# ============================
else:
    raise Exception("UNKOWN USER")

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

AWK_SAVE_LOC = (
    REPO_PATH
    / "pnet_data/processed_files"
    / DATASET_NAME
    / "AwkwardArrs"
    / f"deltaR={MAX_DISTANCE}"
)
NPZ_SAVE_LOC = (
    REPO_PATH
    / "pnet_data/processed_files"
    / DATASET_NAME
    / OUTPUT_DIRECTORY_NAME
    / "SavedNpz"
    / f"deltaR={MAX_DISTANCE}"
    / f"{ENERGY_SCALE=}".lower()
)
