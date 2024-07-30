import numpy as np
import sys
from pathlib import Path
import os
import pandas as pd

REPO_PATH = Path.home() / "workspace/jetpointnet"
SCRIPT_PATH = REPO_PATH / "python_scripts"
sys.path.append(str(SCRIPT_PATH))

from particle import Particle
import numpy as np
from numpy.lib import recfunctions as rfn
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import metrics
from tqdm.auto import tqdm
from numpy.lib import recfunctions as rfn
from jets_training.models.JetPointNet import PointNetSegmentation
from jets_training.jets_train import (
    TRAIN_INPUTS,
    MAX_SAMPLE_LENGTH,
    baseline_configuration,
    EXPERIMENT_NAME,
    NPZ_SAVE_LOC,
    TRAIN_TARGETS,
    TRAIN
)
from data_processing.jets.preprocessing_header import NPZ_SAVE_LOC, POINT_TYPE_ENCODING

#OUTPUT_ACTIVATION_FUNCTION = baseline_configuration['OUTPUT_ACTIVATION_FUNCTION']
#OUTPUT_LAYER_SEGMENTATION_CUTOFF = baseline_configuration['OUTPUT_LAYER_SEGMENTATION_CUTOFF']

NPZ_LOC = NPZ_SAVE_LOC(TRAIN)

PLOT_NON_FOCAL = True
USE_BINARY_ATTRIBUTION_MODEL = True
USE_BINARY_ATTRIBUTION_TRUTH = True
RENDER_IMAGES = True
USE_TRUTH_E = False

# there is an issue that these are needed
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress the specific warning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

NPZ_LOC = NPZ_SAVE_LOC(TRAIN)

def load_data_from_npz(npz_file):
    all_feats = np.load(npz_file)["feats"]
    feats = all_feats[:, :MAX_SAMPLE_LENGTH]  # discard tracking information
    frac_labels = all_feats[:, :MAX_SAMPLE_LENGTH][TRAIN_TARGETS]
    energy_weights = all_feats[:, :MAX_SAMPLE_LENGTH]["cell_E"]
    return feats, frac_labels, energy_weights

def generate_images_and_metadata(sets_to_visualize, model, max_events_per_set, max_images_per_set, save_path):
    metadata_list = []
    SET_UNDER_INVESTIGATION = 'val'

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    TRACK_IMAGE_PATH = save_path / "track_images"
    TRACK_IMAGE_PATH.mkdir(exist_ok=True)
    HIST_PATH = save_path / "meta_results"
    HIST_PATH.mkdir(exist_ok=True)

    for set_name in tqdm(sets_to_visualize):
        print(set_name)
        images_rendered = 0
        events_in_data = 0
        end_set = False
        print(NPZ_LOC / SET_UNDER_INVESTIGATION)
        for file_idx, data_file in tqdm(enumerate(glob.glob(os.path.join(NPZ_LOC / SET_UNDER_INVESTIGATION / set_name, "*.npz"))), leave=False, desc=set_name):
            print(data_file)
            if end_set:
                break
            
            filename_npz = data_file
            feats, fractional_energy, total_cell_energy = load_data_from_npz(filename_npz)

            filtered_features = feats[TRAIN_INPUTS]

            unstructured_filtered_features = rfn.structured_to_unstructured(filtered_features)

            # get prediction
            model_results = model.predict(unstructured_filtered_features)

            # image creation loop
            for window_index, window in enumerate(feats):
                if (events_in_data >= max_events_per_set) and (max_events_per_set !=  -1):
                    end_set = True
                    break
                events_in_data += 1

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
                delta_r_dict = {}
                pt_dict = {}

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
                        cell_total_energy.append(point["cell_E"])
                        cell_model_attribution.append(model_results[window_index][point_index])
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
                                'non_focus_pt': point["track_pt"],
                                'delta_R': point["delta_R"],
                            }
                    elif point['category'] == POINT_TYPE_ENCODING["padding"]:
                        pass
                    else:
                        raise Exception("Unknown point in npz files!")

                cell_truth_attribution = np.array(cell_truth_attribution)
                cell_model_attribution = np.array(cell_model_attribution)


                cell_truth_attribution_cat = np.argmax(rfn.structured_to_unstructured(cell_truth_attribution), axis=-1)
                truth_focal_mask = cell_truth_attribution_cat == 0
                truth_non_focal_mask = cell_truth_attribution_cat == 1
                truth_neutral_mask = cell_truth_attribution_cat == 2


                cell_model_attribution_cat = np.argmax(cell_model_attribution, axis=-1)
                model_focal_mask = cell_model_attribution_cat == 0
                model_non_focal_mask = cell_model_attribution_cat == 1
                model_neutral_mask = cell_model_attribution_cat == 2

                ntruth_focal = np.count_nonzero(cell_truth_attribution_cat == 0)
                ntruth_non_focal = np.count_nonzero(cell_truth_attribution_cat == 1)
                ntruth_neutral = np.count_nonzero(cell_truth_attribution_cat == 2)

                nmodel_focal = np.count_nonzero(cell_model_attribution_cat == 0)
                nmodel_non_focal = np.count_nonzero(cell_model_attribution_cat == 1)
                nmodel_neutral = np.count_nonzero(cell_model_attribution_cat == 2)

                nCells = len(cell_x_list)
                
                valid_cell_mask = window['category'] == 1

                accuracy = np.count_nonzero(cell_model_attribution_cat[valid_cell_mask] == cell_truth_attribution_cat[valid_cell_mask])

                activated_energy = np.sum(window['cell_E'][valid_cell_mask & model_focal_mask])
                ideal_activation_energy = np.sum(window['cell_E'][valid_cell_mask & truth_focal_mask])
                total_truth_track_energy = np.sum(window["truth_cell_total_energy"][valid_cell_mask] * window["truth_cell_focal_fraction_energy"][valid_cell_mask])
                
                # average_energy = np.mean(window["cell_E"])
                # average_truth_energy = np.mean(window["truth_cell_total_energy"][valid_cell_mask] * window["truth_cell_fraction_energy"][valid_cell_mask])

                #tp_rate_partial = tp / n_postive_true_hits
                #fp_rate_partial = fp / nCells
                #tn_rate_partial = tn / n_negative_true_hits
                #fn_rate_partial = fn / nCells

                track_information = {
                    'set_name': set_name,
                    'accuracy': accuracy,
                    'track_ID': window['track_ID'][0],
                    'total_truth_track_E': np.sum(window["truth_cell_total_energy"][valid_cell_mask] * window["truth_cell_fraction_energy"][valid_cell_mask]),
                    'total_real_E': np.sum(window["cell_E"][valid_cell_mask]),
                    'track_pt': window['track_pt'][0],
                    'total_truth_energy': total_truth_track_energy,
                    'activated_energy': activated_energy,
                    'ideal_activation_energy': ideal_activation_energy,
                    'ntruth_focal': ntruth_focal,
                    'ntruth_non_focal': ntruth_non_focal,
                    'ntruth_neutral': ntruth_neutral,
                    'nmodel_focal': nmodel_focal,
                    'nmodel_non_focal': nmodel_non_focal,
                    'nmodel_neutral': nmodel_neutral,
                    'rate_truth_activations': np.sum(cell_truth_attribution == 1)/nCells,
                    'n_non_focal_tracks': len(non_focus_tracks),
                    'n_cells': nCells,
                    'num_correct_predictions': np.sum(cell_truth_attribution == cell_model_attribution),
                    'dumb_accuracy': max(ntruth_focal, nCells-ntruth_focal)/nCells,
                    'positive_dumb_accuracy': ntruth_focal/nCells,
                    'negative_dumb_accuracy': (nCells-ntruth_focal)/nCells,
                    'accuracy': np.sum(cell_truth_attribution == cell_model_attribution)/nCells,
                    # see https://en.wikipedia.org/wiki/Confusion_matrix
                    # for the 4 below, manual implementation has a much faster run time
                    'precision_score': metrics.precision_score(cell_truth_attribution, cell_model_attribution),
                    'recall_score': metrics.recall_score(cell_truth_attribution, cell_model_attribution),
                    'f1_score': metrics.f1_score(cell_truth_attribution, cell_model_attribution),
                    'hamming_loss': metrics.hamming_loss(cell_truth_attribution, cell_model_attribution),
                    # assorted, less useful stats, could delete?
                }

                # model performace

                metadata_list.append(track_information)

                if (RENDER_IMAGES and (max_images_per_set == -1 or images_rendered < max_images_per_set)) or nCells > 1000:
                    images_rendered += 1
                    # convert to plot:
                    fig = plt.figure(figsize=(22, 7))
                    fig.suptitle(f'{set_name}')
                    #fig.suptitle(f'Set: {set_name}, Accuracy factor: {np.sum(cell_truth_attribution == cell_model_attribution)/max(positive, negative):.2f}, Event: {window[0]["event_number"]} Track: {window[0]["track_ID"]}, $\sum E={sum(cell_total_energy):.2f}$, nCells={len(cell_x_list)}, pt={window["track_pt"][0]:.2f}, activated_energy: {float(activated_energy):.4f}, ideal_activation_energy: {ideal_activation_energy:.4f}, truth track E: {total_truth_track_energy:.5f}')

                    ax1 = fig.add_subplot(131, projection='3d')
                    ax2 = fig.add_subplot(132, projection='3d')
                    ax3 = fig.add_subplot(133, projection='3d')

                    ax_list = [ax1, ax2, ax3]

                    for ax_i in ax_list:
                        ax_i.plot(focus_hit_x, focus_hit_y, focus_hit_z, label=name)
                        if PLOT_NON_FOCAL:
                            for non_focal_id, non_focal_track in non_focus_tracks.items():
                                ax_i.plot(non_focal_track['non_focus_hit_x'],
                                        non_focal_track['non_focus_hit_y'], 
                                        non_focal_track['non_focus_hit_z'], 
                                        label=f"Non Focal - ID: {non_focal_id}")
                            ax_i.legend()
                        ax_i.set_xlabel('X Coordinate (mm)')
                        ax_i.set_ylabel('Y Coordinate (mm)')
                        ax_i.set_zlabel('Z Coordinate (mm)')

                    cell_x_array_np = np.array(cell_x_list)
                    cell_y_array_np = np.array(cell_y_list)
                    cell_z_array_np = np.array(cell_z_list)
                    
                    ax1.set_title(f'Model Prediction - {nmodel_focal} | {nmodel_non_focal} | {nmodel_neutral}')
                    ax1.scatter(cell_x_array_np[model_focal_mask], cell_y_array_np[model_focal_mask], cell_z_array_np[model_focal_mask], label="Focal Hits", c='green')
                    ax1.scatter(cell_x_array_np[model_non_focal_mask], cell_y_array_np[model_non_focal_mask], cell_z_array_np[model_non_focal_mask], label="Non Focal Hits", c='yellow')
                    ax1.scatter(cell_x_array_np[model_neutral_mask], cell_y_array_np[model_neutral_mask], cell_z_array_np[model_neutral_mask], label="Neutral Hits", c='red')
                    
                    ax2.set_title(f'Truth Values - {ntruth_focal} | {ntruth_non_focal} | {ntruth_neutral}')
                    ax2.scatter(cell_x_array_np[truth_focal_mask], cell_y_array_np[truth_focal_mask], cell_z_array_np[truth_focal_mask], label="Focal Hits", c='green')
                    ax2.scatter(cell_x_array_np[truth_non_focal_mask], cell_y_array_np[truth_non_focal_mask], cell_z_array_np[truth_non_focal_mask], label="Non Focal Hits", c='yellow')
                    ax2.scatter(cell_x_array_np[truth_neutral_mask], cell_y_array_np[truth_neutral_mask], cell_z_array_np[truth_neutral_mask], label="Neutral Hits", c='red')
                    
                    # Third subplot, total energies

                    if USE_TRUTH_E:
                        ax3.set_title(f'Cell Energies (Truth)')
                    else:
                        ax3.set_title(f'Cell Energies (Not Truth)')
                    
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
                    SET_IMAGE_PATH = TRACK_IMAGE_PATH / set_name
                    SET_IMAGE_PATH.mkdir(exist_ok=True)
                    plt.savefig(SET_IMAGE_PATH / f"event={window[0]['event_number']:09}_track={window[0]['track_ID']:03}.png")
                    plt.close()
                
    metadata = pd.DataFrame(metadata_list)
    metadata.to_csv(HIST_PATH / "metadata.csv")

if __name__ == "__main__":
    #generate_images_and_metadata(sets_to_visualize="", model="", max_events_per_set, max_images_per_set, save_path)
    pass