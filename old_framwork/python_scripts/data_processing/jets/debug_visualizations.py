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


def render_tracks(root_file, NAME='full_plots'):
    for event_index in range(len(root_file['EventTree']['nTrack'].array())):
        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(projection='3d')
        # #ax.set_xlim(-4000,4000)
        # #ax.set_ylim(-7000,7000)
        # #ax.set_zlim(-4000,4000)
        
        # # ax.set_title("ATLAS Radial Calorimeters")
        # ax.set_xlabel('X Coordinate')
        # ax.set_ylabel('Y Coordinate')  # Assuming you meant to plot Z along the vertical in your original plotting order
        # ax.set_zlabel('Z Coordinate')  # And Y on the third dimension
        # plt.tight_layout()


        if root_file['EventTree']['nTrack'].array()[event_index] > 0:
            print('EVENT:', event_index)

            fig = plt.figure(figsize=(20, 12))

            pt_arr = root_file['EventTree']['trackPt'].array()[event_index]

            fig.suptitle(f"Event: {root_file['EventTree']['eventNumber'].array()[event_index]}")

            gs = gridspec.GridSpec(2, 4) 


            # Subplots for different views
            ax1 = fig.add_subplot(gs[0, 0], projection='3d')  # Span two columns
            ax2 = fig.add_subplot(gs[0, 1], projection='3d')  # Span two columns
            ax3 = fig.add_subplot(gs[0, 2], projection='3d')  # Span two columns
            ax4 = fig.add_subplot(gs[0, 3], projection='3d')  # Span two columns
            ax5 = fig.add_subplot(gs[1, 0:2])                   # Span two columns
            ax6 = fig.add_subplot(gs[1, 2:4])                   # Span two columns


            # Set titles and labels
            ax1.set_title('Front View')
            ax2.set_title('Side View')
            ax3.set_title('Top View')
            ax4.set_title('Isometric View')
            ax5.set_title('Eta Space')
            ax6.set_title('Phi Space')

            axl = [ax1, ax2, ax3, ax4]

            for ax in axl:
                ax.set_xlabel('X Coordinate')
                ax.set_ylabel('Y Coordinate')
                ax.set_zlabel('Z Coordinate')
                ax.set_aspect('equal')
        


            track_to_index = root_file['EventTree']['trackTruthParticleIndex'].array()[event_index]
            TruthPdgID = root_file['EventTree']['truthPartPdgId'].array()[event_index]

            eta_arr = root_file['EventTree']['trackEta'].array()[event_index]
            phi_arr = root_file['EventTree']['trackPhi'].array()[event_index]



            spacing = 1
            # Add tracks
            for i in range(root_file["EventTree"]['nTrack'].array()[event_index]):
                print(f"Track {i}")
                xl, yl, zl, etal, phil = [], [], [], [], []
                for loc, rPerp in fixed_r.items():
                    eta = root_file["EventTree"][f'trackEta_{loc}'].array()[event_index][i]
                    phi = root_file["EventTree"][f'trackPhi_{loc}'].array()[event_index][i]
                    if phi > -500:
                        x, y, z = calculate_cartesian_coordinates(eta, phi, rPerp)
                        xl.append(x)
                        yl.append(y)
                        zl.append(z)
                        etal.append(eta)
                        phil.append(phi)

                for loc, z_loc in fixed_z.items():
                    eta = root_file["EventTree"][f'trackEta_{loc}'].array()[event_index][i]
                    phi = root_file["EventTree"][f'trackPhi_{loc}'].array()[event_index][i]
                    if phi > -500:
                        x, y, z = intersection_fixed_z(eta, phi, z_loc)
                        xl.append(x)
                        yl.append(y)
                        zl.append(z)
                        etal.append(eta)
                        phil.append(phi)
                truth_index = track_to_index[i]
                name = Particle.from_pdgid(TruthPdgID[truth_index]).name if truth_index != -1 else "??"
                for ax in axl:
                    ax.plot(xl, yl, zl, label=f"{name} | pt={pt_arr[i]:.2f} | ID={truth_index}")
                next_color = 'red' if truth_index == -1 else random.choice(['blue', 'green', 'purple', 'orange', 'brown', 'olive', 'pink', 'cyan', 'gray'])
                ax5.axvline(eta_arr[i], label=f"{name} | pt={pt_arr[i]:.2f} | ID={truth_index}", color=next_color)
                ax6.axvline(phi_arr[i], label=f"{name} | pt={pt_arr[i]:.2f} | ID={truth_index}", color=next_color)
                    
                #ax5.plot(etal, [spacing] * len(phil), label=f"{name} - {truth_index}")
                #ax6.plot(phil, [spacing] * len(phil), label=f"{name} - {truth_index}")
                spacing += 0.2


            truth_part_eta = root_file['EventTree']['truthPartEta'].array()[event_index]
            truth_part_phi = root_file['EventTree']['truthPartPhi'].array()[event_index]
            truth_pt = root_file['EventTree']['truthPartPt'].array()[event_index]
            truth_E = root_file['EventTree']['truthPartE'].array()[event_index]
            #plot truth data
            for truth_index in range(root_file["EventTree;1"]["nTruthPart"].array()[event_index]):
                #print('TRUTH IDX', truth_index)
                name = Particle.from_pdgid(TruthPdgID[truth_index]).name
                next_color = 'red' if truth_index == -1 else random.choice(['blue', 'green', 'purple', 'orange', 'brown', 'olive', 'pink', 'cyan', 'gray'])
                ax5.axvline(truth_part_eta[truth_index], label=f"TRUTH {name} | pt={truth_pt[truth_index]:.2f} | E={truth_E[truth_index]:.2f} | ID={truth_index}", color=next_color)
                ax6.axvline(truth_part_phi[truth_index], label=f"TRUTH {name} | pt={truth_pt[truth_index]:.2f} | E={truth_E[truth_index]:.2f} | ID={truth_index}", color=next_color)


                
            ax5.set_xlabel('Eta')
            ax5.set_ylabel('Frequency')
            ax6.set_xlabel('Phi')
            ax6.set_ylabel('Frequency')

            eta_arr = geo_file['CellGeo']['cell_geo_eta'].array()
            phi_arr = geo_file['CellGeo']['cell_geo_phi'].array()
            rPerp_arr = geo_file['CellGeo']['cell_geo_rPerp'].array()
            geo_id = geo_file['CellGeo']['cell_geo_ID'].array()
            TruthIndex = root_file['EventTree']['cluster_cell_hitsTruthIndex'].array()[event_index]

            xl, yl, zl = [], [], []
            energy_dep_pid = {}

            # Add Cell Hits
            for idx, cluster_ids in enumerate(root_file["EventTree"]['cluster_cell_ID'].array()[event_index]):
                for cell_idx, cell_id in enumerate(cluster_ids):
                    try:
                        index = ak.where(geo_id == cell_id)

                        truth_index = TruthIndex[idx][cell_idx][0]
                        pdgid = TruthPdgID[truth_index]
                        eta = eta_arr[index]
                        phi = phi_arr[index]
                        rPerp = rPerp_arr[index]
                        x, y, z = calculate_cartesian_coordinates(eta, phi, rPerp)
                        xl.append(x)
                        yl.append(y)
                        zl.append(z)

                        if truth_index in energy_dep_pid.keys():
                            energy_dep_pid[truth_index]['x'].append(x)
                            energy_dep_pid[truth_index]['y'].append(y)
                            energy_dep_pid[truth_index]['z'].append(z)
                            energy_dep_pid[truth_index]['eta'].append(eta[0])
                            energy_dep_pid[truth_index]['phi'].append(phi[0])
                        else:
                            energy_dep_pid[truth_index] = {}
                            energy_dep_pid[truth_index]['x'] = []
                            energy_dep_pid[truth_index]['y'] = []
                            energy_dep_pid[truth_index]['z'] = []
                            energy_dep_pid[truth_index]['eta'] = []
                            energy_dep_pid[truth_index]['phi'] = []
                            energy_dep_pid[truth_index]['pdgId'] = pdgid
                            energy_dep_pid[truth_index]['name'] = Particle.from_pdgid(pdgid).name

                    except Exception as e:
                        print("ERROR ERROR NO CELL ID")
                        print(e)
                        print(cell_id)

                print("Cluster", idx)

            
            for pid, loc_dict in energy_dep_pid.items():
                for ax in axl:
                    ax.scatter(loc_dict['x'], loc_dict['y'], loc_dict['z'], label=f"{loc_dict['name']} | ID={pid}", s=2)
            
            eta_dict = {f"{loc_dict['name']} | ID={pid}": loc_dict['eta'] for pid, loc_dict in energy_dep_pid.items()}
            print(eta_dict.values())
            ax5.hist(eta_dict.values(), label=eta_dict.keys(), histtype='bar', bins=30, stacked=True)

            phi_dict = {f"{loc_dict['name']} | ID={pid}": loc_dict['phi'] for pid, loc_dict in energy_dep_pid.items()}
            ax6.hist(phi_dict.values(), label=phi_dict.keys(), histtype='bar', bins=30, stacked=True)
                
                
            for ax in axl:
                ax.legend()
                ax.scatter(0, 0, 0, label='Origin')
            
            ax5.legend()
            ax6.legend()

            # Set different view angles
            ax1.view_init(elev=90, azim=90)  # Front view
            ax2.view_init(elev=0, azim=90)   # Side view
            ax3.view_init(elev=0, azim=0)    # Top view
            ax4.view_init(elev=30, azim=30)  # Isometric view

            plt.tight_layout()
            plt.savefig(f"root_images/{NAME}/en={root_file['EventTree']['eventNumber'].array()[event_index]}.png", dpi=800)

def viz_awk(ak_array,focal_event_index=0,focal_track_index=0):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    #ax.set_xlim(-4000,4000)
    ##ax.set_ylim(-7000,7000)
    #ax.set_zlim(-4000,4000)
    #ax.set_aspect('equal')
    # ax.set_title("ATLAS Radial Calorimeters")
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Z Coordinate')  
    ax.set_zlabel('Y Coordinate')  
    plt.tight_layout()

    window = ak_array[focal_event_index][focal_track_index]

    plt.title(f"Event: {window['eventNumber']}")

    # add focal track
    ax.plot(list(window['track_layer_intersections']['X']), list(window['track_layer_intersections']['Y']), list(window['track_layer_intersections']['Z']))

    # add adj tracks
    for adj_track in window['associated_tracks']:
        ax.plot(list(adj_track['track_layer_intersections']['X']), list(adj_track['track_layer_intersections']['Y']), list(adj_track['track_layer_intersections']['Z']))

    ax.scatter(list(window['associated_cells']['X']), list(window['associated_cells']['Y']), list(window['associated_cells']['Z']), s=1, marker='.')


    ax.scatter(0, 0, 0, label='Origin')
    plt.legend()
    