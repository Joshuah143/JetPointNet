import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.colors as mcolors
from particle import Particle


def visualize_observed(event: ak.Record, save_name):
    ax = plt.figure().add_subplot(projection="3d")
    ax.set_title("Sample Shower - Simplified Set")
    ax.text(
        0,
        0,
        0,
        f"ATLAS Work in Progress",
        style="italic",
        weight="bold",
        transform=ax.transAxes,
    )

    # plot tracks
    for track in event["tracks"]:
        ax.plot(
            track["hits"]["x"].to_list(),
            track["hits"]["y"].to_list(),
            track["hits"]["z"].to_list(),
        )

    # plot cells
    scatter = ax.scatter(
        event["cells"]["x"].to_list(),
        event["cells"]["y"].to_list(),
        event["cells"]["z"].to_list(),
        c=event["cells"]["cell_E"].to_list(),
        norm=LogNorm(),
        cmap="viridis",
        label="Cells",
        s=2,
    )  #

    plt.colorbar(scatter, ax=ax, label="Cell Energy (Log Scale)")
    ax.legend()
    plt.savefig(save_name, dpi=400, transparent=True)


def visualize_truth(event: ak.Record, save_name, figsize=(5, 5)):
    ax = plt.figure(figsize=figsize).add_subplot(projection="3d")
    ax.set_title("Sample Shower - Simplified Set")
    ax.text(
        0,
        0,
        0,
        f"ATLAS Work in Progress",
        style="italic",
        weight="bold",
        transform=ax.transAxes,
    )
    # set axis names
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # plot tracks
    #:
    # max_pt_track_index = ak.argmax(event['tracks']['truthPartPt'])
    # track = event['tracks'][max_pt_track_index]
    # mcolors.TABLEAU_COLORS
    for track, c in zip(event["tracks"], mcolors.TABLEAU_COLORS):

        truthParticleIndex = track["trackTruthParticleIndex"]

        name = Particle.from_pdgid(track["truthPartPdgId"]).name
        trackPt = track["truthPartPt"]
        name = f"{name} - {trackPt:.2f} GeV"
        # print(f"{name} - {truthParticleIndex} in layer {track['hits']['layer'][-1]} in color {c}")
        ax.plot(
            track["hits"]["x"].to_list(),
            track["hits"]["y"].to_list(),
            track["hits"]["z"].to_list(),
            c=c,
            label=name,
        )
        cells_with_truth = event["cells"][
            ak.fill_none(ak.firsts(event["cells"]["cell_hitsTruthIndex"]), -1)
            == truthParticleIndex
        ]
        ax.scatter(
            cells_with_truth["x"].to_list(),
            cells_with_truth["y"].to_list(),
            cells_with_truth["z"].to_list(),
            c=c,
        )

    # plot cells
    # TODO: plot the cells by largest truth particle
    # I want each cell to be plotted with and in the same
    # color as the track with the same event['cells']['trackTruthParticleIndex'][0] as
    # track[trackTruthParticleIndex]
    # scatter = ax.scatter(event['cells']['x'].to_list(),
    #                      event['cells']['y'].to_list(),
    #                      event['cells']['z'].to_list(),
    #                      c=event['cells']['cell_E'].to_list(),
    #                      norm=LogNorm(),
    #                      cmap='viridis',
    #                      label="Cells", s=2)  #

    # plt.colorbar(scatter, ax=ax, label='Cell Energy (Log Scale)')
    ax.legend()
    plt.savefig(save_name, dpi=400, transparent=True)


def visualize_attributed(event: ak.Record, save_name, figsize=(5, 5)):
    ax = plt.figure(figsize=figsize).add_subplot(projection="3d")
    ax.set_title("Sample Shower - Simplified Set")
    ax.text(
        0,
        0,
        0,
        f"ATLAS Work in Progress",
        style="italic",
        weight="bold",
        transform=ax.transAxes,
    )
    print(event)

    # plot tracks
    for attribution in event["attributed"]:
        track = attribution["track"]
        name = Particle.from_pdgid(track["truthPartPdgId"]).name
        ax.plot(
            track["hits"]["x"].to_list(),
            track["hits"]["y"].to_list(),
            track["hits"]["z"].to_list(),
            label=name,
        )

        # plot cells
        scatter = ax.scatter(
            attribution["cells"]["x"].to_list(),
            attribution["cells"]["y"].to_list(),
            attribution["cells"]["z"].to_list(),
            c=attribution["cells"]["E"].to_list(),
            norm=LogNorm(),
            cmap="viridis",
            label="Cells",
            s=2,
        )
    ax.legend()
