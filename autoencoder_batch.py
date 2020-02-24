#!/usr/bin/python
import argparse

parser = argparse.ArgumentParser(description='Apply NN model in batch regime.')
parser.add_argument('architecture',  metavar='json',   type=str, help='path to the json file with architecture')
parser.add_argument('weights', metavar='h5',   type=str, help='path to the hdf5 file with weights')
parser.add_argument('dataPath',   metavar='path', type=str, help='path to the folder with data')

args = parser.parse_args()

from keras.models import model_from_json
import saxsdocument
import numpy as np
import os
import json

jsonFilename  = args.architecture
h5Filename    = args.weights
inputFolder   = args.dataPath

try:
    # load json and create model
    jsonFile = open(jsonFilename, 'r')
    loadedModelJson = jsonFile.read()
    json_data = json.loads(loadedModelJson)
    smin          = (float)(json_data['smin'])
    smax          = (float)(json_data['smax'])
    firstPointIndex = (int)(json_data['firstPointIndex'])
    lastPointIndex  = (int)(json_data['lastPointIndex'])
    degree        = (float)(json_data['KratkyDegree'])

    jsonFile.close()
    loadedModel = model_from_json(loadedModelJson)
    # load weights into new model
    loadedModel.load_weights(h5Filename)
    inputLength = loadedModel.input_shape[1]  # I(s) points
    print("Expected input: " + str(inputLength) + " points.")
    #outputLength = loadedModel.output_shape[1]  # p(r) points

    print("Model loaded. Yeah!")

except KeyError as e:
    print(f"Error: Oops, model cannot be loaded! Missing value: {e}")
    exit()

except:
    print("Error: Oops, model cannot be loaded for unknown reasons.")
    exit()

Rg = 20.0 # Angstroms

# output csv
outCsv = []
for inputFilename in os.listdir(inputFolder):
    try:
        doc  = saxsdocument.read(os.path.join(inputFolder, inputFilename))
        dat  = np.transpose(np.array(doc.curve[0]))
        s  = dat[0]
        Is = dat[1]

    except Exception as e:
        print(f"Error: Could not read {inputFilename}:")
        print(e)

    if s[0] != 0:
        # sew missing head
        step = s[1] - s[0]
        # find number of missing points
        head_number = (int)(np.rint((s[0] )/step))
        ss = 0.0
        s_head  = np.full(head_number, 0.0)
        Is_head = np.full(head_number, 0.0)
        for i in range(head_number):
            s_head[i]  = ss
            Is_head[i] = np.exp(ss*ss*Rg*Rg/-3.0)
            ss += step
        s  = np.hstack((s_head, s))
        Is = np.hstack((Is_head, Is))

    if len(Is[firstPointIndex:lastPointIndex]) != inputLength:
        print(f"{inputFilename} too short, skipping.")
        continue

    if round(s[firstPointIndex], 3) != round(smin, 3):
        print(f"{inputFilename}: point {firstPointIndex} has s={s[firstPointIndex]}, expected s={smin}")
        exit()

    if round(s[lastPointIndex - 1], 3) != round(smax, 3):
        print(f"{inputFilename}: point {lastPointIndex - 1} has s={s[lastPointIndex]}, expected s={smax}")
        exit()

    Is = Is * s**(degree)
    test = np.array([Is[firstPointIndex:lastPointIndex], ])
    pred = loadedModel.predict(test)
    dat_decoded = np.vstack((s[firstPointIndex:lastPointIndex], pred[0] / s[firstPointIndex:lastPointIndex]**(degree)))
    dat_decoded[1][0] = 1.0  # Make I(0) = 1.0
    np.savetxt(inputFilename, np.transpose(dat_decoded), fmt = "%.8e")
