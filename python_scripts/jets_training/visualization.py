import sys
from pathlib import Path
import os
import matplotlib
import pandas as pd
import seaborn as sns

REPO_PATH = Path.home() / "workspace/jetpointnet"
SCRIPT_PATH = REPO_PATH / "python_scripts"
sys.path.append(str(SCRIPT_PATH))

USER = Path.home().name
if USER == "jhimmens":
    GPU_ID = "1"
elif USER == "luclissa":
    GPU_ID = "0"
else:
    raise Exception("UNKNOWN USER")


if __name__ == "__main__":
    # os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Set GPU

print("Running Visualization Script!")

from particle import Particle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import glob
import math
import time
import pandas
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics
import awkward as ak
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from numpy.lib import recfunctions as rfn
from jets_training.models.JetPointNet import PointNetSegmentation
from jets_training.jets_train import (
    TRAIN_INPUTS,
    MAX_SAMPLE_LENGTH,
    baseline_configuration,
    EXPERIMENT_NAME,
    TRAIN
)
from data_processing.jets.preprocessing_header import NPZ_SAVE_LOC, POINT_TYPE_ENCODING

OUTPUT_ACTIVATION_FUNCTION = baseline_configuration['OUTPUT_ACTIVATION_FUNCTION']
FRACTIONAL_ENERGY_CUTOFF = baseline_configuration['FRACTIONAL_ENERGY_CUTOFF']
OUTPUT_LAYER_SEGMENTATION_CUTOFF = baseline_configuration['OUTPUT_LAYER_SEGMENTATION_CUTOFF']

# there is an issue that these are needed
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress the specific warning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

RESULTS_PATH = REPO_PATH / "results"
RESULTS_PATH.mkdir(exist_ok=True)
MODELS_PATH = REPO_PATH / "models"
MODELS_PATH.mkdir(exist_ok=True)
VISUAL_PATH = REPO_PATH / "visualizations" / f'{EXPERIMENT_NAME}'
VISUAL_PATH.mkdir(exist_ok=True, parents=True)
TRACK_IMAGE_PATH = VISUAL_PATH / "track_images"
TRACK_IMAGE_PATH.mkdir(exist_ok=True)
HIST_PATH = VISUAL_PATH / "meta_results"
HIST_PATH.mkdir(exist_ok=True)

PLOT_NON_FOCAL = True
#TRAIN_DIR = NPZ_SAVE_LOC / "train"
#VAL_DIR = NPZ_SAVE_LOC / "val"
#TEST_DIR = NPZ_SAVE_LOC / "test"
CELLS_PER_TRACK_CUTOFF = 25
USE_BINARY_ATTRIBUTION_MODEL = True
USE_BINARY_ATTRIBUTION_TRUTH = True
RENDER_IMAGES = True
USE_TRUTH_E = False
MAX_FILES = 10
MAX_WINDOWS = 30 # can take -1 for all 
NPZ_LOC = Path("/home/jhimmens/workspace/jetpointnet/pnet_data/processed_files/progressive_training/rho/SavedNpz/deltaR=0.2_maxLen=650/energy_scale=1")
model_path = "/home/jhimmens/workspace/jetpointnet/models/keep/PointNet_best_name=earnest-sweep-1.keras"


print(f"Using: {model_path}")

model = PointNetSegmentation(MAX_SAMPLE_LENGTH, 
                             num_features=len(TRAIN_INPUTS), 
                             num_classes=1, 
                             output_activation_function=OUTPUT_ACTIVATION_FUNCTION)

model.load_weights(model_path)


def load_data_from_npz(npz_file):
    all_feats = np.load(npz_file)["feats"]
    feats = all_feats[:, :MAX_SAMPLE_LENGTH]  # discard tracking information
    frac_labels = all_feats[:, :MAX_SAMPLE_LENGTH]["truth_cell_fraction_energy"]
    energy_weights = all_feats[:, :MAX_SAMPLE_LENGTH]["cell_E"]
    return feats, frac_labels, energy_weights

if USE_BINARY_ATTRIBUTION_MODEL and USE_BINARY_ATTRIBUTION_TRUTH:
    metadata_list = []

images_rendered = 0

for file_idx, data_file in enumerate(glob.glob(os.path.join(NPZ_LOC / 'val', "*.npz"))):
    print(data_file)
    if file_idx > MAX_FILES:
        break
    filename_npz = data_file
    feats, fractional_energy, total_truth_energy = load_data_from_npz(filename_npz)

    filtered_features = feats[TRAIN_INPUTS]

    unstructured_filtered_features = rfn.structured_to_unstructured(filtered_features)

    # get prediction
    model_results = model.predict(unstructured_filtered_features)

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
                if USE_TRUTH_E:
                    cell_total_energy.append(total_truth_energy[window_index][point_index])
                else:
                    cell_total_energy.append(point["cell_E"])
                if USE_BINARY_ATTRIBUTION_MODEL:
                    cell_model_attribution.append(int(model_results[window_index][point_index] >= OUTPUT_LAYER_SEGMENTATION_CUTOFF))
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
                        'non_focus_pt': point["track_pt"],
                        'delta_R': point["delta_R"],
                    }
            elif point['category'] == POINT_TYPE_ENCODING["padding"]:
                pass
            else:
                raise Exception("Unknown point in npz files!")

        cell_truth_attribution = np.array(cell_truth_attribution)
        cell_model_attribution = np.array(cell_model_attribution)

        nCells = len(cell_x_list)

        y_true = (cell_truth_attribution == 1).astype(np.float32)
        y_pred = (cell_model_attribution == 1).astype(np.float32)

        tp = np.sum((cell_model_attribution == 1) & (cell_truth_attribution == 1))
        fp = np.sum((cell_model_attribution == 1) & (cell_truth_attribution == 0))
        tn = np.sum((cell_model_attribution == 0) & (cell_truth_attribution == 0))
        fn = np.sum((cell_model_attribution == 0) & (cell_truth_attribution == 1))
        positive = np.sum(cell_truth_attribution == 1)
        negative = np.sum(cell_truth_attribution == 0)
        predicted_positive = np.sum(cell_model_attribution == 1)
        predicted_negative = np.sum(cell_model_attribution == 0)


        average_energy = np.mean(window["cell_E"])
        average_truth_energy = np.mean(total_truth_energy[window_index])

        #tp_rate_partial = tp / n_postive_true_hits
        #fp_rate_partial = fp / nCells
        #tn_rate_partial = tn / n_negative_true_hits
        #fn_rate_partial = fn / nCells

        track_information = {
            'track_ID': window['track_ID'][0],
            'total_truth_E': sum(total_truth_energy[window_index]),
            'total_real_E': sum(window["cell_E"]),
            'track_pt': window['track_pt'][0],
            'tracks_std_delta_R': 0,
            'tracks_std_pt': 0,
            'truth_attributions': positive,
            'rate_truth_activations': np.sum(cell_truth_attribution == 1)/nCells,
            'model_attributions': predicted_positive,
            'n_non_focal_tracks': len(non_focus_tracks),
            'n_cells': nCells,
            'num_correct_predictions': np.sum(cell_truth_attribution == cell_model_attribution),
            'accuracy': np.sum(cell_truth_attribution == cell_model_attribution)/nCells,
            'average_energy': average_energy,
            'average_truth_energy': average_truth_energy,
            # see https://en.wikipedia.org/wiki/Confusion_matrix
            'num_false_positives': fp,
            'num_false_negatives': fn,
            'num_true_positives': tp,
            'num_true_negatives': tn,
            # for the 4 below, manual implementation has a much faster run time
            'precision_score': metrics.precision_score(y_true, y_pred),
            'recall_score': metrics.recall_score(y_true, y_pred),
            'f1_score': metrics.f1_score(y_true, y_pred),
            'hamming_loss': metrics.hamming_loss(y_true, y_pred),
            # assorted, less useful stats, could delete?
            'false_positive_rate': fp/negative if negative != 0 else 0,
            'false_negative_rate': fn/positive if positive != 0 else 0,
            'true_positive_rate': tp/positive if positive != 0 else 0,
            'true_negative_rate': tn/negative if negative != 0 else 0,
            'positive_predictive_value': tp/predicted_positive if predicted_positive != 0 else 0,
            'false_discovery_rate': fp/predicted_positive if predicted_positive != 0 else 0,
            'false_omission_rate': fn/predicted_negative if predicted_negative != 0 else 0,
            'negative_predictive_value': tn/predicted_negative if predicted_negative != 0 else 0,
            'threat_score': tp/(tp+fn+fp) if tp+fn+fp != 0 else 0,
            #potential to add data from awk files like PDG ID, Chi^2/dof
        }

        # model performace

        metadata_list.append(track_information)

        if (RENDER_IMAGES and (MAX_WINDOWS == -1 or images_rendered < MAX_WINDOWS)) or nCells > 1000:
            images_rendered += 1
            # convert to plot:
            fig = plt.figure(figsize=(22, 7))
            fig.suptitle(f'Event: {window[0]["event_number"]} Track: {window[0]["track_ID"]}, $\sum E={sum(cell_total_energy):.2f}$, nCells={len(cell_x_list)}, pt={window["track_pt"][0]:.2f}')

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
                        print(f"Non Focal - ID: {non_focal_id}")
                    ax_i.legend()
                ax_i.set_xlabel('X Coordinate (mm)')
                ax_i.set_ylabel('Y Coordinate (mm)')
                ax_i.set_zlabel('Z Coordinate (mm)')

            # First subplot
            ax1.set_title(f'Model Prediction - {np.sum(cell_model_attribution)} activations')
            sc1 = ax1.scatter(cell_x_list, cell_y_list, cell_z_list, c=cell_model_attribution, cmap='jet', vmin=0, vmax=1)
            cbar1 = plt.colorbar(sc1, ax=ax1)
            cbar1.set_label('Model Output')
            
            # Second subplot
            ax2.set_title(f'Truth Values - {np.sum(cell_truth_attribution)} activations')
            sc2 = ax2.scatter(cell_x_list, cell_y_list, cell_z_list, c=cell_truth_attribution, cmap='jet', vmin=0, vmax=1)
            cbar2 = plt.colorbar(sc2, ax=ax2)
            cbar2.set_label('frac_E')
            
            # Third subplot, total energies

            if USE_TRUTH_E:
                ax3.set_title(f'Cell Energies (Truth)')
            else:
                ax3.set_title(f'Cell Energies (Not Truth)')
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
            plt.savefig(TRACK_IMAGE_PATH / f"event={window[0]['event_number']:09}_track={window[0]['track_ID']:03}.png")
            matplotlib.pyplot.close()
            
            
# make a 2d hist of truth activations vs model activations
if USE_BINARY_ATTRIBUTION_MODEL and USE_BINARY_ATTRIBUTION_TRUTH:

    metadata = pd.DataFrame(metadata_list)
    metadata.to_csv(HIST_PATH / "metadata.csv")
    total_cells_in_set = sum(metadata['n_cells'])


    number_of_tracks_in_set = len(metadata)

    mask = np.array(metadata['track_pt']) > 3
    mask_name = "pt > 0.5"

    plt.title(f'Metadata Mask Hist')
    plt.hist(metadata['track_pt'], bins=50, label="track_pt")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.yscale('log')
    print('saving cell_rate_truth_activation_hist to', HIST_PATH / 'cell_rate_truth_activation_hist.png')
    plt.savefig(HIST_PATH / 'mask_hist.png', dpi=500)
    plt.clf()

    number_of_tracks_after_mask = np.sum(mask)
    
    print("Generating Hists")

    # might want to try using density=True to make this hists simpler to see


    plt.title(f'Truth to Model activations with nTracks={number_of_tracks_in_set}\n ')
    plt.hist2d(metadata['truth_attributions'], metadata['model_attributions'], bins=50, cmap='viridis', norm=LogNorm()) # range=[x_bounds, y_bounds]
    plt.colorbar(label='Counts')
    plt.xlabel('Truth activations')
    plt.ylabel('Model activations')
    print('saving model_activation_hist to', HIST_PATH / 'raw_model_activation_hist.png')
    plt.savefig(HIST_PATH / 'raw_model_activation_hist.png', dpi=500)
    plt.clf()

    plt.title(f'Truth to Model activations with nTracks={number_of_tracks_after_mask}\n  where {mask_name}')
    plt.hist2d(np.array(metadata['truth_attributions'])[mask], np.array(metadata['model_attributions'])[mask], bins=50, cmap='viridis', norm=LogNorm()) # range=[x_bounds, y_bounds]
    plt.colorbar(label='Counts')
    plt.xlabel('Truth activations')
    plt.ylabel('Model activations')
    print('saving model_activation_hist to', HIST_PATH / 'masked_model_activation_hist.png')
    plt.savefig(HIST_PATH / 'masked_model_activation_hist.png', dpi=500)
    plt.clf()

    plt.title(f'Model activations to number of cells with nTracks={number_of_tracks_in_set}\n ')
    plt.hist2d(metadata['n_cells'], metadata['model_attributions'], bins=50, cmap='viridis', norm=LogNorm()) # range=[x_bounds, y_bounds]
    plt.colorbar(label='Counts')
    plt.xlabel('Number of cells')
    plt.ylabel('Model activations')
    print('saving cell_to_activation_hist to', HIST_PATH / 'cell_to_activation_hist.png')
    plt.savefig(HIST_PATH / 'cell_to_activation_hist.png', dpi=500)
    plt.clf()

    plt.title(f'Model activations to number of cells with nTracks={number_of_tracks_after_mask}\n  where {mask_name}')
    plt.hist2d(np.array(metadata['n_cells'])[mask], np.array(metadata['model_attributions'])[mask], bins=50, cmap='viridis', norm=LogNorm()) # range=[x_bounds, y_bounds]
    plt.colorbar(label='Counts')
    plt.xlabel('Number of cells')
    plt.ylabel('Model activations')
    print('saving cell_to_activation_hist to', HIST_PATH / 'cell_to_activation_hist.png')
    plt.savefig(HIST_PATH / 'cell_to_activation_hist.png', dpi=500)
    plt.clf()

    # Plot the unmasked confusion matrix using seaborn
    tp_unmasked = sum(metadata['num_true_positives'])
    fp_unmasked = sum(metadata['num_false_positives'])
    tn_unmasked = sum(metadata['num_true_negatives'])
    fn_unmasked = sum(metadata['num_false_negatives'])

    print(f"Full unmasked F1 score: {(2 * tp_unmasked)/(2*tp_unmasked+fp_unmasked+fn_unmasked)}")

    confusion_matrix_unmasked = np.array([[tp_unmasked, fp_unmasked], [fn_unmasked, tn_unmasked]])
    
    sns.heatmap(confusion_matrix_unmasked, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Positive', 'Predicted Negative'],
                yticklabels=['Actual Positive', 'Actual Negative'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Unmasked Confusion Matrix')
    plt.savefig(HIST_PATH / 'confusion_matrix.png', dpi=500)
    plt.clf()

    # Plot the masked confusion matrix using seaborn
    tp_masked = sum(np.array(metadata['num_true_positives'])[mask])
    fp_masked = sum(np.array(metadata['num_false_positives'])[mask])
    tn_masked = sum(np.array(metadata['num_true_negatives'])[mask])
    fn_masked = sum(np.array(metadata['num_false_negatives'])[mask])

    print(f"Full masked F1 score: {(2 * tp_masked)/(2*tp_masked+fp_masked+fn_masked)}")

    confusion_matrix_masked = np.array([[tp_masked, fp_masked], [fn_masked, tn_masked]])
    
    sns.heatmap(confusion_matrix_masked, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Positive', 'Predicted Negative'],
                yticklabels=['Actual Positive', 'Actual Negative'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Masked Confusion Matrix where {mask_name}')
    plt.savefig(HIST_PATH / 'confusion_matrix.png', dpi=500)
    plt.clf()


    plt.title(f'Accuracy per track with nTracksMasked={number_of_tracks_after_mask}\n nTracksUnmasked={number_of_tracks_in_set} where {mask_name} ')
    data = metadata['accuracy']
    data_arr = [data, data[mask]]
    labels = ["Unmasked", "Masked"]
    plt.hist(data_arr, bins=50, label=labels, histtype='step', fill=False)
    print('saving cell_accuracy_hist to', HIST_PATH / 'cell_accuracy_hist.png')
    plt.legend()
    plt.savefig(HIST_PATH / 'cell_accuracy_hist.png', dpi=500)
    plt.clf()

    plt.title(f'F1 score per track with nTracks={number_of_tracks_in_set}\n nTracksUnmasked={number_of_tracks_in_set} where {mask_name} ')
    data = metadata['f1_score']
    data_arr = [data, data[mask]]
    labels = ["Unmasked", "Masked"]
    plt.hist(data_arr, bins=50, label=labels, histtype='step', fill=False)
    print('saving cell_f1_hist to', HIST_PATH / 'cell_f1_hist.png')
    plt.legend()
    plt.yscale('log')
    plt.savefig(HIST_PATH / 'cell_f1_hist.png', dpi=500)
    plt.clf()

    plt.title(f'Cell attribution rates per track with nTracks={number_of_tracks_in_set}\n ')
    rates = [metadata['false_positive_rate'], metadata['false_negative_rate'], metadata['true_positive_rate'], metadata['true_negative_rate']]
    labels = ['False Positive Rate', 'False Negative Rate', 'True Positive Rate', 'True Negative Rate']
    plt.hist(rates, label=labels, bins=50, histtype='step', fill=False)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.yscale('log')
    plt.legend()
    print('saving cell_confusion_total_hist to', HIST_PATH / 'cell_confusion_total_hist.png')
    plt.savefig(HIST_PATH / 'cell_confusion_total_hist.png', dpi=500)
    plt.clf()
    
    # Fifth Hist
    plt.title(f'Cell attribution rates per track with nTracks={number_of_tracks_in_set}\n ')
    rates = [metadata['positive_predictive_value'], metadata['false_discovery_rate'], metadata['false_omission_rate'], metadata['negative_predictive_value']]
    labels = ['Positive Predictive Value', 'False Discovery Rate', 'False Omission Rate', 'Negative Predictive Value']
    plt.hist(rates, label=labels, bins=50, histtype='step', fill=False)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.yscale('log')
    plt.legend()
    print('saving cell_predictive_values_hist to', HIST_PATH / 'cell_predictive_values_hist.png')
    plt.savefig(HIST_PATH / 'cell_predictive_values_hist.png', dpi=500)
    plt.clf()


    plt.title(f'Percent Truth Activation with nTracks={number_of_tracks_in_set}\n nTracksUnmasked={number_of_tracks_in_set} where {mask_name} ')
    data = metadata['rate_truth_activations']
    data_arr = [data, data[mask]]
    labels = ["Unmasked", "Masked"]
    plt.hist(data_arr, bins=50, label=labels, histtype='step', fill=False)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.yscale('log')
    print('saving cell_rate_truth_activation_hist to', HIST_PATH / 'cell_rate_truth_activation_hist.png')
    plt.savefig(HIST_PATH / 'cell_rate_truth_activation_hist.png', dpi=500)
    plt.clf()


    print(f"Processed nTracks={number_of_tracks_in_set}")
