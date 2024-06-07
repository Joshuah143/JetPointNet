"""
This script performs Exploratory Data Analysis on pointcloud data. It investigates several data characteristics as:
 (event level)
 - N. events per split
 - N. tracks per event
 - check no split overlap
 
 (sample level, where 'sample' is the set of focus track + associated track + associated cells)
 - N. focus hits per track
 - N. unfocus hits per track
 - N. cell hits per track
 - distribution of each variable (X, Y, Z, Energy)

Author: Luca Clissa <luca.clissa2@unibo.it>
Created: 2024-05-06
License: Apache License 2.0
"""

import sys
from pathlib import Path

REPO_PATH = Path.home() / "workspace/jetpointnet"
SCRIPT_PATH = REPO_PATH / "python_scripts"
sys.path.append(str(SCRIPT_PATH))

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from multiprocessing import Pool

from data_processing.jets.preprocessing_header import (
    AWK_SAVE_LOC,
    NPZ_SAVE_LOC,
    NUM_CHUNK_THREADS,
)
from data_processing.jets.util_functs import get_split
import wandb
import plotly.express as px
import plotly.graph_objects as go

ENERGY_SCALE = 1
DATASET = AWK_SAVE_LOC.parent.parent.name
DELTA_R = float(str(AWK_SAVE_LOC).split("=")[1])

EVENTS_SECTION = "Event Stats"
SAMPLES_SECTION = "Sample Stats"
OVERALL_SECTION = "Total Stats"
SPLIT_SEED, TRAIN_IDS, VAL_IDS, TEST_IDS = get_split(split_seed=62)
MAX_SAMPLE_LENGTH = 278  # NOTE: setting manually for now


def load_data_from_npz(npz_file):
    """Read numpy input files.

    Return:

    feats:
        are the input features

    frac_labels:
        is a bounded value from 0 to 1 representing the amount of truth energy deposited in this cell that belongs to the focused particle

    tot_labels:
        is an unbounded value representing the absolute amount of truth energy deposited in this cell that belongs to the focused particle (in MeV if scaled by 1000 in data preprocessing, GeV if scaled by 1)

    tot_truth_e:
        is the total truth energy deposited into this cell from all particles. I've included it in the dataset for if you want to use during testing

    Labels take the form of a 1D array of size NUM_POINTS, with the index of the output label corresponding to the index of the input point. This is a nice feature of PointNet.
    """
    data = np.load(npz_file)
    feats = data["feats"]  # Shape: (num_samples, MAX_SAMPLE_LENGTH, 6)
    frac_labels = data["frac_labels"]  # Shape: (num_samples, MAX_SAMPLE_LENGTH)
    tot_labels = data["tot_labels"]  # Shape: (num_samples, MAX_SAMPLE_LENGTH)
    tot_truth_e = data[
        "tot_truth_e"
    ]  # Shape: (num_samples, MAX_SAMPLE_LENGTH) (This is the true total energy deposited by particles into this cell)

    return feats, frac_labels, tot_labels, tot_truth_e


def _get_counts_distribution(df, distribution_quantiles):
    counts_distributions_per_split = df.groupby(["split"], observed=True).quantile(
        distribution_quantiles
    )
    counts_distributions_per_split.index.names = ["split", "quantiles"]
    # compute means and std_devs per split
    counts_means_per_split = df.groupby(["split"], observed=True).mean()
    counts_std_dev_per_split = df.groupby(["split"], observed=True).std()

    # Concatenating the means and standard deviations with the quantile distributions
    counts_means_per_split = counts_means_per_split.reset_index()
    counts_means_per_split["quantiles"] = "mean"
    counts_means_per_split.set_index(["split", "quantiles"], inplace=True)

    counts_std_dev_per_split = counts_std_dev_per_split.reset_index()
    counts_std_dev_per_split["quantiles"] = "std_dev"
    counts_std_dev_per_split.set_index(["split", "quantiles"], inplace=True)

    counts_distributions_per_split = pd.concat(
        [
            counts_means_per_split,
            counts_std_dev_per_split,
            counts_distributions_per_split,
        ]
    ).sort_index(level="split", sort_remaining=False)

    counts_distributions_per_split.reset_index(inplace=True)
    counts_distributions_per_split.quantiles = (
        counts_distributions_per_split.quantiles.astype("string")
    )

    # reshape as wide table
    # NOTE: not working when transforming into WB table because of multiindex; this can be handle in WB UI by joining rows based on quantiles
    # _ = counts_distributions_per_split.set_index(["quantiles", "split"])
    # counts_distributions_per_split = _.unstack("split").reset_index()
    return counts_distributions_per_split


def _compute_sample_distribution_stats(sample_data, quantiles):
    feats_means = sample_data.mean(axis=0)
    feats_stds = sample_data.std(axis=0)
    feats_quantiles = np.quantile(sample_data, q=quantiles, axis=0)
    # feats_correlation = np.corrcoef(data_array, rowvar=False)
    # per sample distributions
    sample_stats = np.concatenate(
        (
            feats_means.reshape(1, feats_means.shape[0]),
            feats_stds.reshape(1, feats_stds.shape[0]),
            # feats_means[np.newaxis,:],
            # feats_stds[np.newaxis,:],
            feats_quantiles,
        ),
        axis=0,
    )
    return sample_stats[np.newaxis, :, :]


def _reorder_columns(df):
    ordered_columns = list(df.columns[-3:][::-1]) + list(df.columns[:-3])
    return df[ordered_columns]


def _log_and_save_metadata(
    df, name, wandb_section, save_local_copy=False, local_path=False
):
    df_table = wandb.Table(dataframe=df)
    df_artifact = wandb.Artifact(f"{name}_artifact", type="metadata")
    df_artifact.add(df_table, name)
    if save_local_copy:
        if not local_path:
            warnings.warn(
                f"No `local_path` specified. Saving metadata to {local_path}/{name}.csv",
                UserWarning,
            )
        df_file = local_path / f"{name}.csv"
        local_path.mkdir(exist_ok=True, parents=True)
        df.to_csv(df_file, index=False)
    if local_path:
        df_artifact.add_file(df_file)

    wandb.log({f"{wandb_section}/{name}": df_table})
    wandb.log_artifact(df_artifact)


# Start a run, tracking hyperparameters
wandb.init(
    project="pointcloud",
    config={
        "dataset": DATASET,
        "seed": SPLIT_SEED,
        "delta_R": DELTA_R,
        "energy_scale": ENERGY_SCALE,
    },
    job_type="data-quality",
)

# ============ METADATA ================================================================================

metadata_file = AWK_SAVE_LOC / "metadata" / "hits_per_event.csv"
meta_df = pd.read_csv(metadata_file)
meta_df["split"] = None
meta_df["split"] = pd.Categorical(
    meta_df["split"], categories=["train", "val", "test"], ordered=True
)


for split in ["train", "val", "test"]:
    query = f"eventNumber in @{split.upper()}_IDS"
    split_idxs = meta_df.query(query).index
    meta_df.loc[split_idxs, "split"] = split

# log metadata to wandb
meta_table = wandb.Table(dataframe=meta_df)

# Artifact extends row limit to 200k + easier to re-use
meta_table_artifact = wandb.Artifact("metadata_artifact", type="dataset")
meta_table_artifact.add(meta_table, "metadata_table")
meta_table_artifact.add_file(metadata_file)

wandb.log({f"{OVERALL_SECTION}/metadata": meta_table})
wandb.log_artifact(meta_table_artifact)

# ============ OVERALL STATS ================================================================================

ordered_split_type = pd.api.types.CategoricalDtype(
    categories=["train", "val", "test", "total"], ordered=True
)
n_events = len(meta_df.eventNumber.unique())
n_events_per_split = (
    meta_df.groupby("split", observed=True).eventNumber.unique().apply(len)
)
n_events_per_split["total"] = len(meta_df.eventNumber.unique())
_ = n_events_per_split.reset_index()
_["split"] = _.split.astype(ordered_split_type)
table_events = wandb.Table(
    dataframe=_.sort_values("split"), columns=["split", "N. events"]
)

n_samples_per_split = meta_df.groupby("split", observed=True).trackID.count()
n_samples_per_split["total"] = meta_df.shape[0]
_ = n_samples_per_split.reset_index()
_["split"] = _.split.astype(ordered_split_type)
table_samples = wandb.Table(
    dataframe=_.sort_values("split"), columns=["split", "trackID"]
)

wandb.log(
    {
        f"{OVERALL_SECTION}/n_events": table_events,
        f"{OVERALL_SECTION}/n_samples": table_samples,
    }
)

# wandb.log(
#     {
#         f"{OVERALL_SECTION}/n_events": wandb.plot.bar(
#             table_events, "split", "trackID", title="N. events per split"
#         ),
#         f"{OVERALL_SECTION}/n_samples1": wandb.plot.bar(
#             table_samples, "split", "trackID", title="N. samples per split"
#         ),
#     }
# )

# =======================================================================================================================

# ============ SIZE DISTRIBUTIONS ================================================================================

distribution_quantiles = np.array([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
# distribution_index = ["mean", "std_dev"] + list(distribution_quantiles) - Never used

# tracks X event X split
n_tracks_per_event_per_split = meta_df.groupby(
    ["split", "eventNumber"], observed=True
).trackID.count()

tracks_counts_per_event_per_split = n_tracks_per_event_per_split.groupby(
    "split", observed=True
).value_counts()

df = tracks_counts_per_event_per_split.reset_index()
df["log_count"] = np.log(df["count"])

fig = go.Figure()

splits = df["split"].unique()
for split in splits:
    filtered_df = df.query("split==@split")
    filtered_df.sort_values("trackID", inplace=True)
    fig.add_trace(
        go.Bar(
            x=filtered_df["log_count"],
            y=filtered_df["trackID"],
            name=split,  # this sets the legend and color
            orientation="h",
        )
    )

fig.update_layout(
    barmode="group",
    title="N. tracks per event",
    xaxis_title="log(count)",
    yaxis_title="n_tracks",
    legend_title="split",
    yaxis=dict(autorange="reversed", type="category"),
    legend=dict(
        orientation="h",
        xanchor="center",
        x=0.5,
        y=1.15,
        yanchor="top",
    ),
)

wandb.log({f"{EVENTS_SECTION}/n_tracks_per_event/distribution_plot": wandb.Plotly(fig)})

# # Separate count plots
# for split in ["train", "val", "test"]:
#     tracks_per_event = tracks_counts_per_event_per_split.loc[split].reset_index()
#     tracks_per_event["count"] = np.log(tracks_per_event["count"])
#     tracks_per_event.columns = ["n_tracks", "log(count)"]

#     tracks_per_event_table = wandb.Table(
#         dataframe=tracks_per_event,
#         columns=tracks_per_event.columns,
#     )
#     wandb.log(
#         {
#             f"{EVENTS_SECTION}/n_tracks_per_event/{split}": wandb.plot.bar(
#                 tracks_per_event_table,
#                 "n_tracks",
#                 "log(count)",
#                 title=f"N. tracks per event ({split})",
#             )
#         }
#     )

# compute quantiles per split
n_tracks_per_event_per_split.name = "n_tracks"
n_tracks_distributions_per_split = _get_counts_distribution(
    n_tracks_per_event_per_split, distribution_quantiles
)

n_tracks_distributions_per_split_data = wandb.Table(
    dataframe=n_tracks_distributions_per_split,
    columns=n_tracks_distributions_per_split.columns,
)
wandb.log(
    {
        f"{EVENTS_SECTION}/n_tracks_per_event/distribution_table": n_tracks_distributions_per_split_data
    }
)

# point types X sample X split
count_columns = ["nHits", "nCell", "nUnfocusHits"]
point_type_df = meta_df[["split"] + count_columns].copy()
point_type_df["n_padding"] = MAX_SAMPLE_LENGTH - point_type_df[count_columns].apply(
    sum, axis=1
)


# counts
figures = {}
for point_type in count_columns + ["n_padding"]:
    points = (
        point_type_df[["split", point_type]]
        .value_counts(sort=False)
        .sort_index(ascending=[True, False])
        .reset_index()
    )

    fig = px.bar(
        points,
        x="count",
        y=point_type,
        color="split",
        orientation="h",
        barmode="group",
    )
    fig.show()
    figures[f"{SAMPLES_SECTION}/n_points_per_sample/{point_type}"] = wandb.Plotly(fig)

# boxplots
point_type_counts = px.box(
    point_type_df, y=count_columns + ["n_padding"], color="split", orientation="v"
)

point_type_df[count_columns + ["n_padding"]] = (
    point_type_df[count_columns + ["n_padding"]] / MAX_SAMPLE_LENGTH
)
point_type_pcts = px.box(
    point_type_df, y=count_columns + ["n_padding"], color="split", orientation="v"
)

point_type_counts.update_layout(
    # barmode="group",
    title="Point type per sample",
    xaxis_title="point_type",
    yaxis_title="count",
    legend_title="split",
    # yaxis=dict(autorange="reversed", type="category"),
    legend=dict(
        orientation="h",
        xanchor="center",
        x=0.5,
        y=1.15,
        yanchor="top",
    ),
)

point_type_pcts.update_layout(
    barmode="group",
    title="Point type per sample",
    xaxis_title="point_type",
    yaxis_title="percentage",
    legend_title="split",
    # yaxis=dict(autorange="reversed", type="category"),
    legend=dict(
        orientation="h",
        xanchor="center",
        x=0.5,
        y=1.15,
        yanchor="top",
    ),
)

figures[f"{SAMPLES_SECTION}/n_points_per_sample/distribution_counts"] = wandb.Plotly(
    point_type_counts
)
figures[f"{SAMPLES_SECTION}/n_points_per_sample/distribution_percentages"] = (
    wandb.Plotly(point_type_pcts)
)

wandb.log(figures)

# =======================================================================================================================

# ============ SAMPLE STATS ================================================================================

# tracks X event X split
n_hits_distributions_per_split = _get_counts_distribution(
    meta_df[["split", "nHits", "nCell", "nUnfocusHits"]], distribution_quantiles
)

n_hits_distributions_per_split_data = wandb.Table(
    dataframe=n_hits_distributions_per_split,
    columns=n_hits_distributions_per_split.columns,
)
wandb.log(
    {
        f"{SAMPLES_SECTION}/n_hits_per_event/distribution_table": n_hits_distributions_per_split_data
    }
)
############# old starts here
# for split in ["train", "val"]:
#     stats_df = pd.DataFrame(index=distribution_index)
#     query = f"eventNumber in @{split.upper()}_IDS"
#     split_df = meta_df.query(query)
#     n_tracks_per_event = split_df.groupby(["split", "eventNumber"], observed=True).trackID.count()
#     tracks, counts = np.unique(n_tracks_per_event, return_counts=True)

#     # tracks per event
#     tracks_distributions = n_tracks_per_event.quantile(distribution_quantiles)
#     tracks_distributions.name = "trackCount"
#     tracks_distributions.loc["mean"] = n_tracks_per_event.mean()
#     tracks_distributions.loc["std_dev"] = n_tracks_per_event.std()

#     # focus hits
#     focus_hits = split_df.nHits.quantile(distribution_quantiles)
#     focus_hits.name = "nHits"
#     focus_hits.loc["mean"] = split_df.nHits.mean()
#     focus_hits.loc["std_dev"] = split_df.nHits.std()

#     # associated cells
#     associated_cells = split_df.nCell.quantile(distribution_quantiles)
#     associated_cells.name = "nCell"
#     associated_cells.loc["mean"] = split_df.nCell.mean()
#     associated_cells.loc["std_dev"] = split_df.nCell.std()

#     # associated hits
#     unfocus_hits = split_df.nUnfocusHits.quantile(distribution_quantiles)
#     unfocus_hits.name = "nUnfocusHits"
#     unfocus_hits.loc["mean"] = split_df.nUnfocusHits.mean()
#     unfocus_hits.loc["std_dev"] = split_df.nUnfocusHits.std()

#     stats_df = pd.concat(
#         [stats_df, tracks_distributions, focus_hits, associated_cells, unfocus_hits],
#         axis=1,
#     )

#     # logging
#     n_tracks_per_event_data = [[int(k), np.log(v)] for k, v in zip(tracks, counts)]
#     table = wandb.Table(
#         data=n_tracks_per_event_data, columns=["n_tracks", "log(count)"]
#     )

#     wandb.log(
#         {
#             f"EventStats/{split}/tracks_per_event_counts": wandb.plot.bar(
#                 table, "n_tracks", "log(count)", title="N. track per event"
#             )
#         }
#     )
#     stats_df.insert(0, "quantile", [str(_) for _ in stats_df.index])
#     table = wandb.Table(data=stats_df, columns=stats_df.columns)
#     wandb.log({f"{split}/hits_summary_stats": table})


# for col in ["nHits", "nUnfocusHits", "nCell"]:

#     fig = px.box(meta_df.query("split!='test'"), y=col, x="split", color="split")
#     fig.update_traces(boxmean="sd")
#     fig.update_layout(title=f"{col} distribution")
#     wandb.log({f"Distributions/Points/{col}": wandb.Plotly(fig)})

# =======================================================================================================================


# ============ DISTRIBUTIONS ================================================================================

# variables
N_NUM_FEATS = 5  # categorical features are handled separately
N_NUM_TARGETS = 3
N_DIST_STATS = 2 + len(distribution_quantiles)

TRAIN_DIR = NPZ_SAVE_LOC / "train"
VAL_DIR = NPZ_SAVE_LOC / "val"
# TEST_DIR = NPZ_SAVE_LOC / "test"

sample_features_distributions_df = pd.DataFrame()
sample_targets_distributions_df = pd.DataFrame()
stats_labels = ["mean", "std_dev"] + [str(_) for _ in distribution_quantiles]
for data_dir, split in tqdm(zip([TRAIN_DIR, VAL_DIR], ["train", "val"]), leave=True):
    feature_distributions = np.empty(shape=(0, 7))  # (N_SAMPLES x N_POINTS) X N_FEATS
    target_distributions = np.empty(
        shape=(0, N_NUM_TARGETS)
    )  # (N_SAMPLES x N_POINTS) X N_TARGETS
    sample_feature_distributions = np.empty(
        shape=(0, N_DIST_STATS, N_NUM_FEATS)
    )  # N_SAMPLES X (MEAND + STD + QUANTILES) X N_FEATS
    sample_target_distributions = np.empty(
        shape=(0, N_DIST_STATS, N_NUM_TARGETS)
    )  # N_SAMPLES X (MEAND + STD + QUANTILES) X N_TARGETS
    for fn in tqdm([*data_dir.iterdir()], leave=False, desc=f"{split=}"):
        data = np.load(fn)
        # TODO: implement the same for target columns
        feats = data["feats"]
        frac_labels = data["frac_labels"]  # Shape: (num_samples, MAX_SAMPLE_LENGTH)
        tot_labels = data["tot_labels"]  # Shape: (num_samples, MAX_SAMPLE_LENGTH)
        tot_truth_e = data["tot_truth_e"]
        targets = np.concatenate(
            [
                frac_labels[:, :, np.newaxis],
                tot_labels[:, :, np.newaxis],
                tot_truth_e[:, :, np.newaxis],
            ],
            axis=2,
        )
        # only non-padding points
        hits_mask = feats[:, :, 6] != -1
        target_mask = targets[:, :, 0] != -1

        # TODO: add point-type information
        for idx, sample_hits_mask in enumerate(hits_mask):
            # NOTE: we can't slice directly since we have varying N. of elements along second dimension, so we must iterate one sample at a time
            try:
                sample_hits = feats[idx][sample_hits_mask]
                sample_stats = _compute_sample_distribution_stats(
                    sample_hits[:, 1:-1], distribution_quantiles
                )
                sample_feature_distributions = np.concatenate(
                    (sample_feature_distributions, sample_stats), axis=0
                )
            except IndexError:
                # TODO: debug why this happen: not expected to have empty samples (all padding points) --> perhaps due to events with no tracks?
                pass

            try:
                # NOTE: this may break due to different lengths of targets
                sample_targets_mask = target_mask[idx]
                sample_targets = targets[idx][sample_targets_mask]
                sample_target_stats = _compute_sample_distribution_stats(
                    sample_targets, distribution_quantiles
                )
                sample_target_distributions = np.concatenate(
                    (sample_target_distributions, sample_target_stats)
                )
            except IndexError:
                pass

    dist_df = pd.DataFrame(
        sample_feature_distributions.reshape(-1, N_NUM_FEATS),
        columns=["X", "Y", "Z", "distance_to_track", "E"],
    )
    n_samples = len(dist_df) // len(stats_labels)
    dist_df["quantiles"] = stats_labels * n_samples
    dist_df["sample_id"] = np.repeat(np.arange(n_samples), len(stats_labels))
    dist_df["split"] = split
    sample_features_distributions_df = pd.concat(
        [sample_features_distributions_df, dist_df]
    )

    dist_df = pd.DataFrame(
        sample_target_distributions.reshape(-1, N_NUM_TARGETS),
        columns=["Fraction_Label", "Total_Label", "Total_Truth_Energy"],
    )
    n_samples = len(dist_df) // len(stats_labels)
    dist_df["quantiles"] = stats_labels * n_samples
    dist_df["sample_id"] = np.repeat(np.arange(n_samples), len(stats_labels))
    dist_df["split"] = split
    sample_targets_distributions_df = pd.concat(
        [sample_targets_distributions_df, dist_df]
    )

sample_features_distributions_df = _reorder_columns(sample_features_distributions_df)
sample_targets_distributions_df = _reorder_columns(sample_targets_distributions_df)


_log_and_save_metadata(
    sample_features_distributions_df,
    name="sample_features_distributions",
    wandb_section=f"{SAMPLES_SECTION}/distributions/features",
    save_local_copy=True,
    local_path=NPZ_SAVE_LOC / "metadata",
)
_log_and_save_metadata(
    sample_targets_distributions_df,
    name="sample_targets_distributions",
    wandb_section=f"{SAMPLES_SECTION}/distributions/targets",
    save_local_copy=True,
    local_path=NPZ_SAVE_LOC / "metadata",
)

# split-level distributions
sample_features_distributions_df["quantiles"] = pd.Categorical(
    sample_features_distributions_df["quantiles"], categories=stats_labels, ordered=True
)
split_features_distributions_df = (
    sample_features_distributions_df.groupby(["split", "quantiles"], observed=True)
    .mean()
    .iloc[:, 1:]
    .reset_index()
)

sample_targets_distributions_df["quantiles"] = pd.Categorical(
    sample_targets_distributions_df["quantiles"], categories=stats_labels, ordered=True
)
split_targets_distributions_df = (
    sample_targets_distributions_df.groupby(["split", "quantiles"], observed=True)
    .mean()
    .iloc[:, 1:]
    .reset_index()
)

split_features_distributions_table = wandb.Table(
    dataframe=split_features_distributions_df
)
split_targets_distributions_table = wandb.Table(
    dataframe=split_targets_distributions_df
)

wandb.log(
    {
        f"{SAMPLES_SECTION}/distributions/features/split_features_distributions": split_features_distributions_table,
        f"{SAMPLES_SECTION}/distributions/targets/split_targets_distributions": split_targets_distributions_table,
    }
)


wandb.finish()


# ### old starts here

# # # overall distributions
# # feats.reshape(-1, feats.shape[-1]).shape
# # feature_distributions = np.concatenate(
# #     (feature_distributions, feats.reshape(-1, feats.shape[-1])), axis=0
# # )
# split = "train"

# point_type_labels = {0: "cell", 1: "focus hit", 2: "unfocus hit", -1: "padding"}
# feature_distributions = pd.DataFrame(
#     feature_distributions,
#     columns="eventNumber X Y Z distance_to_track E point_type".split(" "),
# )

# # point type counts
# point_type_data = feature_distributions.point_type.value_counts()
# point_type_data.index = point_type_data.index.map(point_type_labels)
# table = wandb.Table(
#     data=[
#         [label, np.log(point_type_data.loc["padding"])]
#         for label in point_type_data.index
#     ],
#     columns=["point_type", "log(count)"],
# )
# wandb.log(
#     {
#         f"{split}/overall_point_type_counts": wandb.plot.bar(
#             table, "point_type", "log(count)", title="N. points per type"
#         )
#     }
# )


# # distributions per point type
# feature_distributions = feature_distributions.query("point_type != -1")
# feature_distributions["point_type_label"] = (
#     feature_distributions["point_type"].map(point_type_labels).astype("category")
# )
# fig = go.Figure()

# # Configuration for side-by-side placement
# groups = feature_distributions["point_type_label"].unique()
# width = 0.2  # Box width
# shift = [-0.2, 0, 0.2]  # Shift for each variable

# for idx, col in enumerate(["X", "Y", "Z"]):
#     fig.add_trace(
#         go.Box(
#             y=feature_distributions[col],
#             x=[
#                 i + shift[idx]
#                 for i in feature_distributions["point_type_label"].cat.codes
#             ],  # Adjust positions
#             name=col,
#             boxpoints="outliers",  # Show only outliers as individual points
#             width=width,
#             offsetgroup=col,
#             alignmentgroup=col,
#             boxmean="sd",
#         )
#     )

# # Update layout for better visualization
# fig.update_layout(
#     title="Cartesian coordinates by point type",
#     xaxis=dict(
#         title="Point Type",
#         tickmode="array",
#         tickvals=list(range(len(groups))),
#         ticktext=groups,
#     ),
#     yaxis_title="Values",
#     legend_title="Variable",
#     boxmode="group",  # group bars of same x position
# )

# # Show plot
# fig.show()


# for feat in feature_distributions.columns[1:-1]:
#     break
#     wandb.log(
#         {
#             f"{split}/{feat}": feature_distributions.query("point_type != -1")[
#                 feat
#             ].quantile(distribution_quantiles)
#         }
#     )


# feature_distributions.groupby(["eventNumber", "trackID"]).point_type

# wandb.log(
#     {
#         f"{split}/": feature_distributions.query("point_type != -1")[feat].quantile(
#             distribution_quantiles
#         )
#     }
# )

# # =======================================================================================================================

# BATCH_SIZE = 4096
# TRAIN_DIR = NPZ_SAVE_LOC / "train"
# VAL_DIR = NPZ_SAVE_LOC / "val"
# # TEST_DIR = NPZ_SAVE_LOC / "test"

# data_dir = TRAIN_DIR
# batch_size = BATCH_SIZE

# fn = [*TRAIN_DIR.iterdir()][0]
# data = np.load(fn)
# feats = data["feats"]
# feats.shape
