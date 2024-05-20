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
add_tracks_as_labels = False
NUM_EVENTS_PER_CHUNK = 200
TRAIN_SPLIT_RATIO = 0.55
VAL_SPLIT_RATIO = 0.3
# TEST_SPLIT_RATIO is implied to be the remaining percentage
NUM_THREAD_PER_CHUNK = 25  # root to awk
NUM_CHUNK_THREADS = 16  # awk to npz
# OUTPUT_DIRECTORY_NAME = "rho_full/"
OUTPUT_DIRECTORY_NAME = "ttbar/"
# ============================


DEBUG_NUM_EVENTS_TO_USE = None
UPROOT_MASK_VALUE_THRESHOLD = -100_000
MAX_DISTANCE = 0.1

# Path to the ROOT file containing jet events

# RHO DATA
# FILE_LOC = "/eos/home-m/mswiatlo/images/truthPerCell/rho_full.root"
# GEO_FILE_LOC = "/eos/home-m/mswiatlo/images/truthPerCell/cell_geo.root"

# JETS DATA
# FILE_LOC = "/data/atlas/mltree_2000_fixedHits.root"
# GEO_FILE_LOC = "/data/atlas/data/rho_delta/rho_small.root"
FILE_LOC = "/eos/home-m/mswiatlo/forLuca/mltree_large.root"
GEO_FILE_LOC = "/eos/home-m/mswiatlo/images/truthPerCell/cell_geo.root"

AWK_SAVE_LOC = (
    REPO_PATH
    / "pnet_data/processed_files"
    / OUTPUT_DIRECTORY_NAME
    / "AwkwardArrs"
    / f"deltaR={MAX_DISTANCE}"
)
NPZ_SAVE_LOC = (
    REPO_PATH
    / "pnet_data/processed_files"
    / OUTPUT_DIRECTORY_NAME
    / "SavedNpz"
    / f"deltaR={MAX_DISTANCE}"
)
