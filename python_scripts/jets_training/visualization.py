import sys
from pathlib import Path
import os
import matplotlib
import pandas as pd

REPO_PATH = Path.home() / "workspace/jetpointnet"
SCRIPT_PATH = REPO_PATH / "python_scripts"
sys.path.append(str(SCRIPT_PATH))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set GPU - Must be done before importing tf

print("Running Visualization Script!")

from particle import Particle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import glob
import math
import time
import awkward as ak
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from numpy.lib import recfunctions as rfn
from jets_training.models.JetPointNet import PointNetSegmentation
from jets_training.jets_train import (
    TRAIN_INPUTS,
    MAX_SAMPLE_LENGTH,
    OUTPUT_ACTIVATION_FUNCTION,
    BATCH_SIZE,
    FRACTIONAL_ENERGY_CUTOFF,
    best_checkpoint_path,
    last_checkpoint_path
)
from data_processing.jets.preprocessing_header import NPZ_SAVE_LOC, AWK_SAVE_LOC
from data_processing.jets.util_functs import POINT_TYPE_ENCODING

RESULTS_PATH = REPO_PATH / "results"
RESULTS_PATH.mkdir(exist_ok=True)
MODELS_PATH = REPO_PATH / "models"
MODELS_PATH.mkdir(exist_ok=True)
VISUAL_PATH = REPO_PATH / "visualizations"
VISUAL_PATH.mkdir(exist_ok=True)

RESTRICT_RANGE = False
PLOT_NON_FOCAL = True
EPOCHS = 10 # back to marco's settings
TRAIN_DIR = NPZ_SAVE_LOC / "train"
VAL_DIR = NPZ_SAVE_LOC / "val"
CELLS_PER_TRACK_CUTOFF = 25
USE_BINARY_ATTRIBUTION_MODEL = True
USE_BINARY_ATTRIBUTION_TRUTH = True
RENDER_IMAGES = True
MAX_WINDOWS = -1

model_path = last_checkpoint_path # best_checkpoint_path

model = PointNetSegmentation(MAX_SAMPLE_LENGTH, 
                             num_features=len(TRAIN_INPUTS), 
                             num_classes=1, 
                             output_activation_function=OUTPUT_ACTIVATION_FUNCTION)

model.load_weights(model_path)

def load_data_from_npz(npz_file):
    data = np.load(npz_file)
    feats = data['feats'][:, :MAX_SAMPLE_LENGTH]  # Shape: (num_samples, 859, 6)
    frac_labels = data['frac_labels'][:, :MAX_SAMPLE_LENGTH]  # Shape: (num_samples, 859)
    tot_labels = data['tot_labels'][:, :MAX_SAMPLE_LENGTH]  # Shape: (num_samples, 859)
    tot_truth_e = data['tot_truth_e'][:, :MAX_SAMPLE_LENGTH]  # Shape: (num_samples, 859) (This is the true total energy deposited by particles into this cell)
    return feats, frac_labels, tot_labels, tot_truth_e

filename_npz = VAL_DIR / "chunk_0_val.npz"
feats, fractional_energy, _, total_truth_energy = load_data_from_npz(filename_npz)

filtered_features = feats[TRAIN_INPUTS]

unstructured_filtered_features = rfn.structured_to_unstructured(filtered_features)
# get prediction
model_results = model.predict(unstructured_filtered_features)
# segmentation_logits = np.squeeze(segmentation_logits, axis=-1)

# make a 2d hist of truth activations vs model activations
if USE_BINARY_ATTRIBUTION_MODEL and USE_BINARY_ATTRIBUTION_TRUTH:
    model_attributions = []
    truth_attributions = []
    n_cells = []


# image creation loop
for window_index, window in enumerate(feats):

    name = f'Focal Track - ID: {window[0]["track_ID"]}' # add PDGID name??

    focus_hit_x = []
    focus_hit_y = []
    focus_hit_z = []

    cell_x_list = []
    cell_y_list = []
    cell_z_list = []
    cell_model_attribution = []
    cell_truth_attribution = []
    cell_total_energy = []

    non_focus_tracks = {}

    for point_index, point in enumerate(window):

        # this would be much nicer as a match-case statement, but it kept giving me issues
        if point['category'] == POINT_TYPE_ENCODING["focus hit"]:
            focus_hit_x.append(point["normalized_x"])
            focus_hit_y.append(point["normalized_y"])
            focus_hit_z.append(point["normalized_z"])
        elif point['category'] == POINT_TYPE_ENCODING["cell"]:
            cell_x_list.append(point["normalized_x"])
            cell_y_list.append(point["normalized_y"])
            cell_z_list.append(point["normalized_z"])
            cell_total_energy.append(total_truth_energy[window_index][point_index])
            if USE_BINARY_ATTRIBUTION_MODEL:
                cell_model_attribution.append(int(model_results[window_index][point_index] >= FRACTIONAL_ENERGY_CUTOFF))
            else:
                cell_model_attribution.append(model_results[window_index][point_index])
            if USE_BINARY_ATTRIBUTION_TRUTH:
                cell_truth_attribution.append(int(fractional_energy[window_index][point_index] >= FRACTIONAL_ENERGY_CUTOFF))
            else:
                cell_truth_attribution.append(fractional_energy[window_index][point_index])
        elif point['category'] == POINT_TYPE_ENCODING["unfocus hit"]:
            if point['track_ID'] in non_focus_tracks:
                non_focus_tracks[point['track_ID']]['non_focus_hit_x'].append(point["normalized_x"])
                non_focus_tracks[point['track_ID']]['non_focus_hit_y'].append(point["normalized_y"])
                non_focus_tracks[point['track_ID']]['non_focus_hit_z'].append(point["normalized_z"])
            else:
                non_focus_tracks[point['track_ID']] = {
                    'non_focus_hit_x': [point["normalized_x"]],
                    'non_focus_hit_y': [point["normalized_y"]],
                    'non_focus_hit_z': [point["normalized_z"]],
                }
        elif point['category'] == POINT_TYPE_ENCODING["padding"]:
            pass
        else:
            raise Exception("Unknown point in npz files!")

    n_cells.append(len(cell_x_list))
    model_attributions.append(cell_model_attribution.count(1))
    truth_attributions.append(cell_model_attribution.count(1))

    if RENDER_IMAGES:
        # convert to plot:
        fig = plt.figure(figsize=(22, 7))
        fig.suptitle(f'Event: {window[0]["event_number"]} Track: {window[0]["track_ID"]}, $\sum E={sum(cell_total_energy)}$')

        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')

        ax_list = [ax1, ax2, ax3]

        for ax_i in ax_list:
            ax_i.plot(focus_hit_x, focus_hit_y, focus_hit_z, label=name)
            if PLOT_NON_FOCAL:
                for non_focal_id, non_focal_track in non_focus_tracks.items():
                    ax_i.plot(non_focal_track['non_focus_hit_x'],
                            non_focal_track['non_focus_hit_x'], 
                            non_focal_track['non_focus_hit_x'], 
                            label=f"Non Focal - ID: {non_focal_id}")
                ax_i.legend()
            ax_i.set_xlabel('X Coordinate (mm)')
            ax_i.set_ylabel('Y Coordinate (mm)')
            ax_i.set_zlabel('Z Coordinate (mm)')
            

        # First subplot
        ax1.set_title('Model Prediction')
        sc1 = ax1.scatter(cell_x_list, cell_y_list, cell_z_list, c=cell_model_attribution, cmap='jet', vmin=0, vmax=1)
        cbar1 = plt.colorbar(sc1, ax=ax1)
        cbar1.set_label('Model Output')
        
        # Second subplot
        sc2 = ax2.scatter(cell_x_list, cell_y_list, cell_z_list, c=cell_truth_attribution, cmap='jet', vmin=0, vmax=1)
        cbar2 = plt.colorbar(sc2, ax=ax2)
        cbar2.set_label('frac_E')
        
        # Third subplot, total energies

        cell_x_array_np = np.array(cell_x_list)
        cell_y_array_np = np.array(cell_y_list)
        cell_z_array_np = np.array(cell_z_list)
        cell_total_energy_array = np.array(cell_total_energy)

        mask = np.array(cell_total_energy) > 0

        # Ensure normalization is consistent with LogNorm only for positive values
        if np.any(mask):
            sc3 = ax3.scatter(cell_x_array_np[mask], cell_y_array_np[mask], cell_z_array_np[mask], c=cell_total_energy_array[mask], cmap='jet', norm=LogNorm())
        else:
            sc3 = ax3.scatter(cell_x_array_np, cell_y_array_np, cell_z_array_np, cmap='jet')

        cbar3 = plt.colorbar(sc3, ax=ax3)
        cbar3.set_label('Total_Label (MeV)')

        plt.tight_layout()
        print(f"saving: event={window[0]['event_number']:09}_track={window[0]['track_ID']:03}")
        plt.savefig(f"visualizations/event={window[0]['event_number']:09}_track={window[0]['track_ID']:03}.png")
        matplotlib.pyplot.close()
        if (window_index > MAX_WINDOWS) and (MAX_WINDOWS != -1):
            break
            


# make a 2d hist of truth activations vs model activations
if USE_BINARY_ATTRIBUTION_MODEL and USE_BINARY_ATTRIBUTION_TRUTH:
    #x_bounds = (min(flat_model), max(flat_model))  # min and max for x-axis
    #y_bounds = (0, 1)  # min and max for y-axis

    print("Generating Hist")
    plt.hist2d(truth_attributions, model_attributions, bins=50, cmap='viridis', norm=LogNorm()) # range=[x_bounds, y_bounds]
    plt.colorbar(label='Counts')

    # Labels and title
    plt.xlabel('Truth activations ')
    plt.ylabel('Model activations')
    plt.title(f'Truth to Model activations with n={sum(n_cells)}')

    # Save the plot to a file
    plt.savefig(f'visualizations/model_activation_hist.png', dpi=500)
    plt.clf()














# DEPRECATED DO NOT USED BELOW:
"""   

# get source from awk
ak_file = "/home/jhimmens/workspace/jetpointnet/pnet_data/processed_files/2000_events_w_fixed_hits/AwkwardArrs/deltaR=0.1/train/chunk_0_train.parquet"
ak_array = ak.from_arrow(pq.read_table(ak_file))


# print(segmentation_logits[0]) # predictions from 1st events
# [print(i) for i in segmentation_logits[1]]
# all non-0 bar,

array_model_outputs = np.array([])
array_real_fracs = np.array([])
array_cell_energies = np.array([])

# number of elements in the set is the max, in this case train chunk 0
segmentation_offset = 0
print(f"{len(ak_array)} Events Found")
for event_number in range(3):
    event = ak_array[event_number]
    offset = 0
    sample_idx = 0
    tracks_in_event = len(event)
    while (sample_idx + offset < tracks_in_event):
        try:
            while len(event[sample_idx + offset]["associated_cells"]) < CELLS_PER_TRACK_CUTOFF:
                offset += 1
                print(f"Offset at: {event_number}, {sample_idx}, {offset}")
                # it gets stuck here for some reason
        except IndexError:
            continue

        # i think these two lines might not be needed
        if not (sample_idx + offset < tracks_in_event - 1):
            continue

        ak_track = event[sample_idx + offset]
        
        
        print(f"-----------{sample_idx}-------------")
        
        sample_features = feats[sample_idx + segmentation_offset]
        energies = tot_labels[sample_idx + segmentation_offset]  # not used anywhere - look at documentation before useing
        fracs = frac_labels[sample_idx + segmentation_offset]
        tot_true_energy = tot_truth_e[sample_idx + segmentation_offset] # total energies for particles, can apply a cut with this

        energy_point_indices = sample_features[:, 6] == 1 # ensure the first track interactions and other data are ignored
        print(f"numer of energy: {len([i for i in energy_point_indices if i == 1])}")

        # predicted_classes = np.where(segmentation_logits[sample_idx] > 0, 1, 0) # changed to greater than 0.5
        true_classes = np.where(frac_labels[sample_idx + segmentation_offset] > 0.5, 1, 1) # changed both conditions to 1 so that the dimensionallity works

        energies_filtered = tot_truth_e[sample_idx + segmentation_offset][energy_point_indices]
        # predicted_classes_filtered = predicted_classes[energy_point_indices]
        filtered_predicted_raw = segmentation_logits[sample_idx + segmentation_offset][energy_point_indices]
        filtered_input_fracs = fracs[true_classes[energy_point_indices]]

        print(f"{filtered_input_fracs.shape=}")

        print(f"{len(tot_true_energy[energy_point_indices])=}")

        print(f"REAL = {len(filtered_predicted_raw)}")

        print(f"{len(ak_track['associated_cells'])=}")

        print(f"{len(ak_track['associated_tracks'])=}")

        print(f'{offset=}')

        print(f'{sample_idx=}')

        true_classes_filtered = true_classes[energy_point_indices]

        chibydof = ak_track['trackChiSquared/trackNumberDOF']
    

        name = "Focal Track" #= str(ak_track['track_part_Idx']) # Particle.from_pdgid(ak_array[0][0]['track_part_Idx']).name

        # plot non-focal tracks
        
        if PLOT_NON_FOCAL:
            x_non_focal = []
            y_non_focal = []
            z_non_focal = []
            for non_f_track in ak_track['associated_tracks']:
                xn = []
                yn = []
                zn = []
                for layer in non_f_track['track_layer_intersections']:
                    xn.append(layer['X'])
                    yn.append(layer['Y'])
                    zn.append(layer['Z'])
                x_non_focal.append(xn)
                y_non_focal.append(yn)
                z_non_focal.append(zn)
                

        xl, yl, zl = [], [], []
        for i in ak_track['track_layer_intersections']:
            xl.append(i['X'])
            yl.append(i['Y'])
            zl.append(i['Z'])
        

        xs = [j['X'] for j in event[sample_idx + offset]['associated_cells']]
        ys = [j['Y'] for j in event[sample_idx + offset]['associated_cells']]
        zs = [j['Z'] for j in event[sample_idx + offset]['associated_cells']]

        # Assuming filtered_predicted_raw and frac_eng are arrays of the same length as associated_cells
        filtered_predicted_raw = np.array(filtered_predicted_raw)
        # frac_eng = filtered_input_fracs
        frac_eng = np.array([j['Fraction_Label'] for j in ak_track['associated_cells']])
        total_label = np.array([j['Total_Label'] for j in ak_track['associated_cells']])


        # Set up the figure and subplots
        fig = plt.figure(figsize=(22, 7))

        x_range = (min(xs), max(xs))
        y_range = (min(ys), max(ys))
        z_range = (min(zs), max(zs))


        p_raw = filtered_predicted_raw

        fig.suptitle(' '.join(['$\\frac{\\chi^2}{dof}$:', f'{chibydof:.2f},', f'Event: {ak_track["eventNumber"]}, Track ID: {ak_track["track_ID"]}, $\\sum E$: {np.sum(total_label):.3f} MeV']), fontsize=16)

        # First subplot
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(xl, yl, zl, label=name)
        if PLOT_NON_FOCAL:
            for x, y, z in zip(x_non_focal, y_non_focal, z_non_focal):
                ax1.plot(z,y,z, label="Non Focal")
            ax1.legend()
        ax1.set_xlabel('X Coordinate (mm)')
        ax1.set_ylabel('Y Coordinate (mm)')
        ax1.set_zlabel('Z Coordinate (mm)')
        ax1.set_title('Raw model outut')
        array_model_outputs = np.append(array_model_outputs, np.array(p_raw))
        sc1 = ax1.scatter(xs, ys, zs, c=p_raw, cmap='jet')
        cbar1 = plt.colorbar(sc1, ax=ax1)
        cbar1.set_label('Model Output')
        if RESTRICT_RANGE:
            ax1.set_xlim(x_range)
            ax1.set_ylim(y_range)
            ax1.set_zlim(z_range)

        # Second subplot
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot(xl, yl, zl, label=name)
        if PLOT_NON_FOCAL:
            for x, y, z in zip(x_non_focal, y_non_focal, z_non_focal):
                ax2.plot(z,y,z, label="Non Focal")
            ax2.legend()
        ax2.set_xlabel('X Coordinate (mm)')
        ax2.set_ylabel('Y Coordinate (mm)')
        ax2.set_zlabel('Z Coordinate (mm)')
        ax2.set_title('Focal track deposit on cell')
        array_real_fracs = np.append(array_real_fracs, np.array(frac_eng))
        sc2 = ax2.scatter(xs, ys, zs, c=frac_eng, cmap='jet')
        cbar2 = plt.colorbar(sc2, ax=ax2)
        cbar2.set_label('frac_E')
        if RESTRICT_RANGE:
            ax2.set_xlim(x_range)
            ax2.set_ylim(y_range)
            ax2.set_zlim(z_range)

        # Third subplot, total energies
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.plot(xl, yl, zl, label=name)
        if PLOT_NON_FOCAL:
            for x, y, z in zip(x_non_focal, y_non_focal, z_non_focal):
                ax3.plot(z,y,z, label="Non Focal")
            ax3.legend()
        ax3.set_xlabel('X Coordinate (mm)')
        ax3.set_ylabel('Y Coordinate (mm)')
        ax3.set_zlabel('Z Coordinate (mm)')
        ax3.set_title('Energy per Cell for E > 0')

        array_cell_energies = np.append(array_cell_energies, np.array(total_label))

        mask = np.array(total_label) > 0

        # Ensure normalization is consistent with LogNorm only for positive values
        if np.any(mask):  # Ensure there's at least one positive value
            # Ensure normalization is consistent with LogNorm only for positive values
            sc3 = ax3.scatter(np.array(xs)[mask], np.array(ys)[mask], np.array(zs)[mask], c=total_label[mask], cmap='jet', norm=LogNorm())
        else:
            # Use linear scale if no positive values are present
            sc3 = ax3.scatter(np.array(xs)[mask], np.array(ys)[mask], np.array(zs)[mask], c=total_label[mask], cmap='jet')

        cbar3 = plt.colorbar(sc3, ax=ax3)
        cbar3.set_label('Total_Label (MeV)')
        if RESTRICT_RANGE:
            ax3.set_xlim(x_range)
            ax3.set_ylim(y_range)
            ax3.set_zlim(z_range)

        # Adjust layout
        plt.tight_layout()
        plt.savefig(f'images/event={ak_track["eventNumber"]}_track={ak_track["track_ID"]}.png')
        matplotlib.pyplot.close()
        sample_idx += 1
    segmentation_offset += sample_idx

# histogram of energies

flat_energy = array_cell_energies.flatten()

plt.hist(flat_energy, bins=50, log=True, range=(0, 5))
plt.yscale('log')

# Labels and title
plt.xlabel('Energy (MeV)')
plt.ylabel('Count (log scale)')
plt.title('Cell Energies')

# Save the plot to a file
plt.savefig('1d_histogram_log_scale.png', dpi=500)
plt.clf()

E_MASK_VAL = 0.01

for i in [0, 0.0001, 0.01, 0.1, 0.5, 0.8, 1, 5]:
    E_MASK_VAL = i

    energy_mask = flat_energy >= E_MASK_VAL

    flat_model = array_model_outputs.flatten()[energy_mask]
    flat_truth = array_real_fracs.flatten()[energy_mask]


    x_bounds = (min(flat_model), max(flat_model))  # min and max for x-axis
    y_bounds = (0, 1)  # min and max for y-axis

    print("Generating Hist")
    plt.hist2d(flat_model, flat_truth, bins=50, cmap='viridis', range=[x_bounds, y_bounds], norm=LogNorm())
    plt.plot([0,1], [0,1], label="Ideal")
    plt.colorbar(label='Counts')
    plt.legend()

    # Labels and title
    plt.xlabel('Model Outputs')
    plt.ylabel('Real Fractions')
    plt.title(f'2D Histogram of Model Outputs vs Real Fractions;\n Cell E > {E_MASK_VAL} MeV')

    # Save the plot to a file
    plt.savefig(f'2d_histogram_e_{E_MASK_VAL}.png', dpi=500)
    plt.clf()

"""