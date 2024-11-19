import uproot
import numpy as np
import awkward as ak
import pandas as pd
from pathlib import Path
from pprint import pprint
from uproot.extras import pandas
from visualizations.np_visualization import visualize_sample

from to_numpy import event_to_trainable

def main(config):
    example_set = ak.from_json(Path("/Users/jhimmens/Library/CloudStorage/Dropbox/Work/TRIUMF/jetpointnet/prod/test_input_JZ4.json"))
    #print(event_to_trainable(example_set)[0][0])
    #print(event_to_trainable(example_set)[0])
    print((example_set).show(type=True))

    for i in range(len(example_set)):
        event = example_set[i]
        if len(event['tracks']) == 0:
            continue
        print("IMPORTATING PART LOOK HERE")
        print(event)
        trainable_event = event_to_trainable(event)
        print(trainable_event[0])
        print(trainable_event.dtype)
        print(trainable_event)
        visualize_sample(trainable_event, f'views/sample_training_JZ4_{i}.png')
    # df = pd.array(trainable

if __name__ == "__main__":
    main({})