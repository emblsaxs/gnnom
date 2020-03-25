#!/usr/bin/python
import argparse
parser = argparse.ArgumentParser(description='Apply NN model in batch regime.')
parser.add_argument('architecture',  metavar='json',   type=str, help='path to the json file with architecture')
parser.add_argument('weights', metavar='h5',   type=str, help='path to the hdf5 file with weights')
parser.add_argument('dataPath',   metavar='path', type=str, help='path to the folder with data')
parser.add_argument('-p', '--prefix', type=str, default='', help='prefix for the output file names')
args = parser.parse_args()
from keras.models import model_from_json, Model
import saxsdocument
import numpy as np
import os
import json
jsonFilename  = args.architecture
h5Filename    = args.weights
inputFolder   = args.dataPath
prefix = args.prefix
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
    numberOfPoints = (lastPointIndex - firstPointIndex)
    print("Model loaded. Yeah!")
    # get intermediate output from the hidden layer
    layerName = "dense_1"
    hiddenLayerModel = Model(inputs = loadedModel.input, output = loadedModel.get_layer(layerName).output)
except KeyError as e:
    print(f"Error: Oops, model cannot be loaded! Missing value: {e}")
    exit()
except Exception as e:
    print(f"Error: Oops, model cannot be loaded: {e}.")
    exit()

# output pdb
outPdb = ""
for num, inputFilename in enumerate(os.listdir(inputFolder)):
    try:
        prop, cur = saxsdocument.read(os.path.join(inputFolder, inputFilename))
        inputS = cur['s']
        Is = cur['I']
    except Exception as e:
        print(f"Error: Could not read {inputFilename}:")
        print(e)

    test = np.array([Is, ])
    pred = loadedModel.predict(test)
    classifierPred = hiddenLayerModel.predict(test)
    x = 100.0*pred[0][0]
    y = 100.0*pred[0][1]
    z = 100.0*pred[0][2]
    j = []
    j.append("ATOM".ljust(6))  # atom#6s
    j.append(str(num + 1).rjust(5))  # aomnum#5d
    j.append("CA".center(4))  # atomname$#4s
    j.append("THR".ljust(3))  # resname#1s
    j.append("B".rjust(1))  # Astring
    j.append("1".rjust(4))  # resnum
    j.append(('%8.3f' % (float(x))).rjust(8))  # x
    j.append(('%8.3f' % (float(y))).rjust(8))  # y
    j.append(('%8.3f' % (float(z))).rjust(8))  # z\
    j.append(str('%6.2f' % (float(1.00))).rjust(6))  # occ
    j.append(str('%6.2f' % (float(100.09))).ljust(6))  # temp
    j.append("C".rjust(12))  # elname
    outPdb += f"{j[0]}{j[1]} {j[2]} {j[3]} {j[4]}{j[5]}    {j[6]}{j[7]}{j[8]}{j[9]}{j[10]}{j[11]}\n"
    #outPdb += ("%s%s %s %s %s%s    %s%s%s%s%s%s\n" % j[0], j[1], j[2], j[3], j[4], j[5], j[6], j[7], j[8], j[9], j[10],j[11])
    #outPdb += f"ATOM     {num}  CA  THR B   2      {x:.3f}  {y:.3f}  {z:.3f}  1.00100.09           C  \n"
    dat_decoded = np.vstack((inputS, pred[0]))
    np.savetxt(os.path.join(prefix,inputFilename), np.transpose(dat_decoded), fmt = "%.8e", header = "Creator: autoencoder")

with open("class3d.pdb", "w") as file:
    file.write(outPdb)
#np.savetxt(os.path.join(prefix,"class3d.pdb"), outPdb, fmt = "%s", header = "Creator: autoencoder")