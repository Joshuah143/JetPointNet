import awkward as ak

try:
    import tensorflow.keras as keras
except ImportError:
    import keras


def apply_reduction_array(array: ak.Array, model: keras.Model):
    # Loop until, no tracks are present
    # to numpy
    # predict
    # apply
    while ak.max(ak.num(array["tracks"])) > 0:
        array = array[ak.num(array["tracks"]) > 0]
        event_np = ak.to_numpy(array)
        event_np = event_np.reshape(1, *event_np.shape)
        pred = model.predict(event_np)
        array["attributed"]["rho"] = pred[0]


def apply_reduction_record(record: ak.Record, model: keras.Model):
    event_np = ak.to_numpy(record)
    event_np = event_np.reshape(1, *event_np.shape)
