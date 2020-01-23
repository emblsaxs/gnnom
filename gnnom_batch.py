#!/usr/bin/python
import argparse

parser = argparse.ArgumentParser(description='Apply NN model in batch regime.')
parser.add_argument('architecture',  metavar='json',   type=str, help='path to the json file with architecture')
parser.add_argument('weights', metavar='h5',   type=str, help='path to the hdf5 file with weights')
parser.add_argument('dataPath',   metavar='path', type=str, help='path to the folder with data')
parser.add_argument('--first', type=int, default=1,  help='index of the first point to use (default: 1)')
parser.add_argument('--last', help='index of the last point to use (default: use all)')
parser.add_argument('--smax',  type=float, default=0.0, help='Smax')
parser.add_argument('-o', '--output', type=str, default="", help='save output in CSV format')

args = parser.parse_args()

from keras.models import model_from_json
import saxsdocument
import numpy as np
import os
import json

jsonFilename  = args.architecture
h5Filename    = args.weights
inputFolder   = args.dataPath
outCsvPath    = args.output
firstPointIndex = int(args.first) - 1
stdpddf       = 1.0

try:
    # load json and create model
    jsonFile = open(jsonFilename, 'r')
    loadedModelJson = jsonFile.read()
    json_data = json.loads(loadedModelJson)
    if('Normalization coefficient' in json_data):
        stdpddf = (float)(json_data['Normalization coefficient'])
    jsonFile.close()
    loadedModel = model_from_json(loadedModelJson)
    # load weights into new model
    loadedModel.load_weights(h5Filename)
    inputLength = loadedModel.input_shape[1]  # I(s) points
    print("Expected input: " + str(inputLength) + " points.")
    #outputLength = loadedModel.output_shape[1]  # p(r) points


#    if int(args.last) == -1:
#        lastPointIndex = inputLength
#    else:
#        lastPointIndex = int(args.last)

    print("Model loaded. Yeah!")
except Exception as e:
    print("Error: Oops, model can not be uploaded.")
    print(e)
    exit()

Rg = 20.0 # Angstroms

# output csv
if args.last is not None: args.last = (int)(args.last)
outCsv = []
for inputFilename in os.listdir(inputFolder):
    try:
        print(inputFilename)
        doc  = saxsdocument.read(os.path.join(inputFolder, inputFilename))
        dat  = np.transpose(np.array(doc.curve[0]))
        s  = dat[0][firstPointIndex:args.last]
        Is = dat[1][firstPointIndex:args.last]
        if (args.last is not None) and (len(Is) != args.last - firstPointIndex):
            print(f"{inputFilename} length is wrong. Expected: {inputLength}; Received: {len(Is)}.")
            continue
    except Exception as e:
        print("Error: Could not read input data")
        print(e)

    if args.smax > 0:
        if s[-1] < args.smax:
            print(f"{inputFilename} Smax is less than {args.smax}.")
            continue

    if s[0] != 0:
        # sew missing head
        step = s[1] - s[0]
        # find number of missing points

        head_number = (int)(np.rint((s[0] )/step))
        ss = 0.0
        Is_head = np.full(head_number, 0.0)
        for i in range(head_number):
            Is_head[i] = np.exp(ss*ss*Rg*Rg/-3.0)
            ss += step
        Is = np.hstack((Is_head, Is))

    test = np.array([Is, ])
    pred = loadedModel.predict(test)

    #TODO: instead of checking output number of points > 10 read model type (scalar/pddf)
    if len(pred[0]) > 10:
        # Find Dmax: first negative point after max(p(r))
        max_pddf = np.argmax(pred)
        negIndex = np.argmax(pred[:,max_pddf:] < 0)
        # Crop p(r > Dmax), nullify last point
        pred = pred[:, 0: (negIndex + max_pddf + 1)]
        pred[:,-1] = 0.0

        r = np.arange(0.0, len(pred[0]) * 0.25, 0.25)
        outCsv.append(inputFilename + ', ' + str(round(r[-1], 3)))
        # print(f"{len(r)} - {len(pred[0])} - {r[-1]}") # DEBUG
        pddf_predicted = np.vstack((r, stdpddf * pred[0]))
        np.savetxt(inputFilename, np.transpose(pddf_predicted), fmt = "%.8e")

    else:
        for number in pred[0]:
            outCsv.append(inputFilename + ', ' + str(round(number, 3)))

if outCsvPath != "":
    np.savetxt(outCsvPath, outCsv, delimiter=",", fmt='%s')
    print(outCsvPath + " is written.")
else:
    print ("\n".join(outCsv))
