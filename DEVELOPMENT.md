# Delta R masking

Delta R calculations have been added into the npz files, but the cut is still happening at the `root_to_awk` level. This masking could be deleted (line 661 and 825 `util_functs`) and a mask could be applied at the `jets_train` level as a hyper parameter. However, this would change the max length caculations very significantly as removing the mask would include all event data and the filtering would be non-trivial due to the structure of the npz files. 

Delta R fr tracks is defined by thier interaction with EMB2, for the delta R that gets applied the tracks should this be the case.

# Bugs

- Line 626 util functs: all associeted tracks are given the label of `1`. This seems like a mistake, but it doesnt really matter since we dont carry it through to the training data.

- Very low priority, but the awk arrays have really hard to read naming with a mix of Id, ID, and other confusing names

- `LR` gets overwritten in `jets_train` because of the `CustomLRScheduler`

- If the chunk size is small then is increased, the files orignially created will not be overwritten causing data duplication


