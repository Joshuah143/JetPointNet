# Delta R masking

Delta R calculations have been added into the npz files, but the cut is still happening at the `root_to_awk` level. This masking could be deleted (line 661 and 825 `util_functs`) and a mask could be applied at the `jets_train` level as a hyper parameter. However, this would change the max length caculations very significantly as removing the mask would include all event data and the filtering would be non-trivial due to the structure of the npz files. 

Delta R fr tracks is defined by their interaction with EMB2, for the delta R that gets applied the tracks should this be the case.

# Negative Cell Energy

TODO: Negative cell energy and its effects on `frac_label` deserve further exploration.

# Todo: (Joshua)


- Switch to the right Wandb project
- Use Wandb like you are meant to 
- Investigate Negative Energies
- Add Delta R cut to train
- Add to distribution file, get it to work
- Implement custom data load class
- *Investigate shuffling effects training*
- Implement Hyper Parameter Search
- Investigate Focal BCE loss function
- Convert to simple `.fit()`
- Move features to a meta_data array so that we can do comparative analysis of included vs excluded points and of the train, val, and test sets.

# Bugs

- Line 626 util functs: all associated tracks are given the label of `1`. This seems like a mistake, but it doesn't really matter since we dont carry it through to the training data.

- Very low priority, but the awk arrays have really hard to read naming with a mix of Id, ID, and other confusing names

- If the chunk size is small then is increased, the files originally created will not be overwritten causing data duplication

- 4 arrays are saved to the npz files, only 3 of them are ever used and only 2 are actually relevant to the model, could be low hanging fruit for speed improvement.

- `NUM_CHUNK_THREADS` is always used by awk_to_npz, but we could actually use the `min(NUM_CHUNK_THREADS, num_chunks)` to avoid empty processes

- There is a results path that we define in `jets_train` but never use: 
```
RESULTS_PATH = REPO_PATH / "result" / EXPERIMENT_NAME
RESULTS_PATH.mkdir(exist_ok=True, parents=True)
```

In any case, I added it to the `.gitignore` file.

- Should the following be included in the model filename?

```
MAX_SAMPLE_LENGTH, 
num_features=len(TRAIN_INPUTS), 
num_classes=1, 
output_activation_function=OUTPUT_ACTIVATION_FUNCTION,
```

- `ENERGY_SCALE` should be included in `visualization.py` so that the units on graphs work


# Grid Notes

Input dataset: https://bigpanda.cern.ch/files/?datasetid=551156493

prun --exec "bash_wrapper.sh %IN" --inDS user.mswiatlo.801167.Py8EG_A14NNPDF23LO_jj_JZ2.recon.ESD.e8514_e8528_s4185_s4114_r14977_2024.06.14.1_mltree.root.551156493.551156493 --outDS user.jhimmens.`uuidgen` --outputs "*.npz" --nFilesPerJob 5 --noBuild --nJobs 3 --extFile ./python_scripts/data_processing/geo_data/cell_geo.root

other things:
--output myout.txt
--nJobs 5 - will this just run all? will it loop? max cuncurrent? on my attempt it still ran ~950 jobs

Is there a home directory, since we depend on one?
Is there a way to only take successful files or filter the incoming set?

cp /eos/home-m/mswiatlo/images/truthPerCell/cell_geo.root /eos/user/j/jhimmens/cell_geo.root

### copy over files to eos using scp
scp -r /home/jhimmens/workspace/jetpointnet/pnet_data/processed_files/attempt_1_june_18/full_set/SavedNpz jhimmens@lxplus.cern.ch:/eos/user/j/jhimmens/jetpointnet/data/attempt_1_june_18/full_set/
 
### using rsync
rsync -avz --delete /home/jhimmens/workspace/jetpointnet/pnet_data/processed_files/attempt_1_june_18/full_set/SavedNpz jhimmens@lxplus.cern.ch:/eos/user/j/jhimmens/jetpointnet/data/attempt_1_june_18/full_set/


# notes
file prefixes like `user.mswiatlo.39955678` are constent for each JZ set
