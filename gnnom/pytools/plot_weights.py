"""
Extracts and plots the weights from a given layer of a NN.
Useful to estimate how many weights are used, how many repetative weights are present and how noisy they are.
To smooth the weights use the regularization parameter in the makemodel_scalar.py script.
Save weights in *.int format to be compatible with primus Qt.
"""

import argparse

parser = argparse.ArgumentParser(description='Plot NN weights.')
parser.add_argument('weights', metavar='h5', type=str, help='path to the hdf5 file with weights')
parser.add_argument('--json', metavar='json', type=str, help='path to the json file with architecture')
parser.add_argument('layer', metavar='l', type=int, help='number of layer to plot')

args = parser.parse_args()

import numpy as np
import os
import json
from keras.models import model_from_json

h5Filename = args.weights
jsonFilename = args.json
# load json assuming the same base filename for json and h5 files!
if jsonFilename is None: jsonFilename = os.path.splitext(h5Filename)[0] + ".json"
l = args.layer

try:
    jsonFile = open(jsonFilename, 'r')
    loadedModelJson = jsonFile.read()
    json_data = json.loads(loadedModelJson)
    smin = float(json_data['smin'])
    smax = float(json_data['smax'])
    firstPointIndex = int(json_data['firstPointIndex'])
    lastPointIndex = int(json_data['lastPointIndex'])

    jsonFile.close()
    loadedModel = model_from_json(loadedModelJson)
    # load weights into new model
    loadedModel.load_weights(h5Filename)
    # loadedModel.summary()
    weightsLayer = loadedModel.layers[l].get_weights()[0]
    # weightsLayer = loadedModel.weights[l]
    print(f"Number of weights in {l} layer is : {weightsLayer.shape}")

except KeyError as e:
    print(f"Error: Oops, model cannot be loaded! Missing value: {e}")
    exit()

except Exception as e:
    print(f"Error: {e}")
    exit()

numberOfPoints = (lastPointIndex - firstPointIndex)
step = smax / (numberOfPoints - 1)
s = np.arange(0.0, smax + 0.0000001, step)
w = np.transpose(weightsLayer)
print(f"s: {s.shape} w: {w.shape}")
name = loadedModel.name + '_0_.int'
np.savetxt(name, np.transpose(np.vstack((s, w))), fmt="%.8e")
print(f"{name} file is written.")
