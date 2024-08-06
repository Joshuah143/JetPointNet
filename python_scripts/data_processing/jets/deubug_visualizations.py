# these are sample functions meant to be copied into a notebook and modified for debugging, npz visualization shoudl be done in the training set
import uproot
import awkward as ak
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
from numpy.lib import recfunctions as rfn

REPO_PATH = Path.home() / "workspace/jetpointnet"
SCRIPT_PATH = REPO_PATH / "python_scripts"
sys.path.append(str(SCRIPT_PATH))

from data_processing.jets.common_utils import calculate_cartesian_coordinates, intersection_fixed_z
from data_processing.jets.track_metadata import fixed_r, fixed_z
from data_processing.jets.preprocessing_header import POINT_TYPE_LABELS, POINT_TYPE_ENCODING


GEO_FILE_LOC = "/fast_scratch_1/atlas/pflow/rho_small.root"

# likely root awkward arr:
awk_file = "/fast_scratch_1/atlas/pflow/jhimmens_working_files/pnet_data/processed_files/collected_data/rev_3/AwkwardArrs/deltaR=0.2/val/JZ4/user.mswiatlo.39955735._000005.mltree.root_chunk_0_val.parquet"
likely_root_file = "/fast_scratch_1/atlas/pflow/20240614/user.mswiatlo.801169.Py8EG_A14NNPDF23LO_jj_JZ4.recon.ESD.e8514_e8528_s4185_s4114_r14977_2024.06.14.1_mltree.root/user.mswiatlo.39955735._000005.mltree.root"

root_file = uproot.open(likely_root_file)
geo_file = uproot.open(GEO_FILE_LOC)
root_file["EventTree"].keys()


def viz_root(event_tree, geo_tree, event_id):
    focal_index = event_id
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    #ax.set_xlim(-4000,4000)
    #ax.set_ylim(-7000,7000)
    #ax.set_zlim(-4000,4000)
    ax.set_aspect('equal')
    # ax.set_title("ATLAS Radial Calorimeters")
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')  # Assuming you meant to plot Z along the vertical in your original plotting order
    ax.set_zlabel('Z Coordinate')  # And Y on the third dimension
    plt.tight_layout()
    plt.title(f"Event: {root_file['EventTree']['eventNumber'].array()[focal_index]}")
    full_phi_list = [] 
    trig = False
    # Add tracks
    for i in range(len(root_file["EventTree"]['trackPhi_EMB1'].array()[focal_index])):
        xl, yl, zl = [], [], []
        for loc, rPerp in fixed_r.items():
            eta = root_file["EventTree"][f'trackEta_{loc}'].array()[focal_index][i]
            phi = root_file["EventTree"][f'trackPhi_{loc}'].array()[focal_index][i]
            if phi > -500:
                print(phi)
                full_phi_list.append(phi)
                x, y, z = calculate_cartesian_coordinates(eta, phi, rPerp)
                xl.append(x)
                yl.append(y)
                zl.append(z)
        for loc, z_loc in fixed_z.items():
            eta = root_file["EventTree"][f'trackEta_{loc}'].array()[focal_index][i]
            phi = root_file["EventTree"][f'trackPhi_{loc}'].array()[focal_index][i]
            if phi > -500:
                print(phi)
                full_phi_list.append(phi)
                x, y, z = intersection_fixed_z(eta, phi, z_loc)
                xl.append(x)
                yl.append(y)
                zl.append(z)
        if len(xl)>0 and trig == False:
            print(i)
            ax.plot(xl, yl, zl, "--", label=f"{i} track", linewidth=10)
            print(xl, yl, zl)
            trig = True
        else:
            ax.plot(xl, yl, zl, "--",)

    eta_arr = geo_file['CellGeo']['cell_geo_eta'].array()
    phi_arr = geo_file['CellGeo']['cell_geo_phi'].array()
    rPerp_arr = geo_file['CellGeo']['cell_geo_rPerp'].array()
    xl, yl, zl = [], [], []

    # Add Cell Hits
    for idx, cluster_ids in enumerate(root_file["EventTree"]['cluster_cell_ID'].array()[focal_index]):
        if True:
            for cell_id in cluster_ids:
                index = ak.where(geo_file['CellGeo']['cell_geo_ID'].array() == cell_id)
                eta = eta_arr[index]
                phi = phi_arr[index]
                rPerp = rPerp_arr[index]
                x, y, z = calculate_cartesian_coordinates(eta, phi, rPerp)
                xl.append(x)
                yl.append(y)
                zl.append(z)
        print(idx)


    ax.scatter(xl, yl, zl, s=1, marker='.')
    te = 0
    ax.scatter(xl[te], yl[te], zl[te], label=f"CELL ID {root_file['EventTree']['cluster_cell_ID'].array()[focal_index][0][te]}")

    ax.scatter(0, 0, 0, label='Origin')
    plt.legend()
    plt.show()

def viz_npz(npz_array):
    pass