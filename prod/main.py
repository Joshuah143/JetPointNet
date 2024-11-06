import uproot
import numpy as np
import awkward as ak
import pandas as pd
from pathlib import Path
from pprint import pprint
from uproot.extras import pandas

from to_numpy import event_to_trainable

def main(config):
    example_set = ak.from_json(Path("test_inputjz4.json"))
    #print(event_to_trainable(example_set)[0][0])
    #print(event_to_trainable(example_set)[0])
    trainable_events = event_to_trainable(example_set)
    print("1" * 30)
    print(trainable_events.dtype)
    print(trainable_events[0])
    # df = pd.array(trainable_events[0][0])
    # print(df)
    # print(np.array2string(trainable_events[0][0]))


if __name__ == '__main__':
    main({})
