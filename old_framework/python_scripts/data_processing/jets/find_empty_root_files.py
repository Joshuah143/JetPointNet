import glob
import json
import os
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
from tqdm.auto import tqdm

REPO_PATH = Path.home() / "workspace/jetpointnet"
SCRIPT_PATH = REPO_PATH / "python_scripts"
sys.path.append(str(SCRIPT_PATH))


from data_processing.jets.awk_utils import *
from data_processing.jets.common_utils import *
from data_processing.jets.preprocessing_header import *

root_files = glob.glob(os.path.join(ROOT_FILES_DIR, "**/*.root"), recursive=True)
print(len(root_files))
empty = []
for file in root_files:
    if os.path.isfile(file):
        with uproot.open(file) as root:
            if len(root.keys()) == 0:
                empty.append(file)

with open("empty_roots.json", "w") as f:
    json.dump(empty, f, indent="\t")

print(len(empty))
