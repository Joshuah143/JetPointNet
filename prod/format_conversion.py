import awkward as ak
import numpy as np

event_array_dtype = np.dtype([
    ('event_number', np.int32),
    ('cell_ID', np.int32),
    ('track_ID', np.int32),


    # -1 if non-truth event
    ('truth_cell_focal_fraction_energy', np.float32),
    ('truth_cell_non_focal_fraction_energy', np.float32),
    ('truth_cell_neutral_fraction_energy', np.float32),
    ('truth_cell_total_energy', np.float32),

    ('category', np.int8),
    ('track_num', np.int32),
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('delta_R', np.float32),
    ('eta', np.float32),
    ('phi', np.float32),
    # ('distance', np.float32), was included... is it still required? Hard to calculate
    ('normalized_x', np.float32),
    ('normalized_y', np.float32),
    ('normalized_z', np.float32),
    # ('normalized_distance', np.float32), see above
    ('cell_sigma', np.float32),
    ('track_chi2_dof', np.float32),
    # ("track_chi2_dof_cell_sigma", np.float32), TBD if combined classes are required/useful
    ('cell_E', np.float32),
    ('normalized_cell_E', np.float32),
    ('track_pt', np.float32),
    ('normalized_track_pt', np.float32),
    # ('track_pt_cell_E', np.float32), TBD if combined classes are required/useful
    # ('normalized_track_pt_cell_E', np.float32), TBD if combined classes are required/useful
])

def event_to_trainable(event_ak: ak.Array, delta_R=0.2, truth=True, max_event_len=800) -> np.ndarray:
    # Choose highest pT track

    # create array

    # add highest pT track

    # add tracks within delta R

    # normalize E, pT, x, y, z