# Delta R masking

Delta R calculations have been added into the npz files, but the cut is still happening at the `root_to_awk` level. This masking could be deleted (line 661 and 825 `util_functs`) and a mask could be applied at the `jets_train` level as a hyper parameter. However, this would change the max length caculations very significantly as removing the mask would include all event data and the filtering would be non-trivial due to the structure of the npz files. 

Delta R fr tracks is defined by their interaction with EMB2, for the delta R that gets applied the tracks should this be the case.

# Bugs

- Line 626 util functs: all associated tracks are given the label of `1`. This seems like a mistake, but it doesn't really matter since we dont carry it through to the training data.

- Very low priority, but the awk arrays have really hard to read naming with a mix of Id, ID, and other confusing names

- `LR` gets overwritten in `jets_train` because of the `CustomLRScheduler`

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
