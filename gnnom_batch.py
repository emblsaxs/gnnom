"""
Apply NN for prediction.
"""
import argparse

parser = argparse.ArgumentParser(description='Apply NN model in batch regime.')
parser.add_argument('architecture',  metavar='json',   type=str, help='path to the json file with architecture')
parser.add_argument('weights', metavar='h5',   type=str, help='path to the hdf5 file with weights')
parser.add_argument('dataPath', metavar='path', type=str, help='path to the folder with data')
parser.add_argument('-o', '--output', type=str, default="", help='prefix to output CSV files')

args = parser.parse_args()

from keras.models import model_from_json
import saxsdocument
import numpy as np
import os
import json
from normalisation.logarithm import normalise  # , unnormalise

jsonFilename  = args.architecture
h5Filename    = args.weights
inputFolder   = args.dataPath
outCsvPath    = args.output
stdpddf       = 1.0

try:
    # load json and create model
    jsonFile = open(jsonFilename, 'r')
    loadedModelJson = jsonFile.read()
    json_data = json.loads(loadedModelJson)
    # Optional fields in json
    if 'Normalization coefficient' in json_data:
        stdpddf = float(json_data['Normalization coefficient'])
    if 'meanIs' in json_data:
        meanIs = json_data['meanIs']
        stdIs = json_data['stdIs']
    elif 'meanIs' not in json_data:
        print(f"ACHTUNG! "
              f"{jsonFilename} does not contain normalization coefficients!"
              f"Proceeding without normalization...")
    # Compulsory fields in json
    smin = (float)(json_data['smin'])
    smax = (float)(json_data['smax'])
    firstPointIndex = (int)(json_data['firstPointIndex'])
    lastPointIndex = (int)(json_data['lastPointIndex'])

    jsonFile.close()
    loadedModel = model_from_json(loadedModelJson)
    # load weights into new model
    loadedModel.load_weights(h5Filename)
    inputLength = loadedModel.input_shape[1]  # I(s) points
    print(f"Expected input: {inputLength} points.")
    # outputLength = loadedModel.output_shape[1]  # p(r) points
    print("Model loaded. Yeah!")

except KeyError as e:
    print(f"Error: Oops, model cannot be loaded! Missing value: {e}")
    quit()

except Exception as e:
    print(f"Error: {e}")
    quit()

# Rg = 20.0 # Angstroms


dataFiles = os.listdir(args.dataPath)
dataFiles.sort()
folders = ["dat-c025", "dat-c05", "dat-c1", "dat-c2", "dat-c4", "dat-c8", "dat-c16"]

for f in folders:
    outCsv = []
    print(f"Processing folder: {f}")
    t = os.path.join(args.dataPath, "test", f)
    dataFiles = os.listdir(t)
    dataFiles.sort()
    for inputFilename in dataFiles:
        try:
            cur, __ = saxsdocument.read(os.path.join(t, inputFilename))
            Is = cur['I'][firstPointIndex: lastPointIndex + 1]

        except Exception as e:
            print(f"Error: Could not read {inputFilename}:")
            print(e)
            continue

        # if s[0] != 0:
        #    # sew missing head
        #    step = s[1] - s[0]
        #    # find number of missing points
        #    head_number = (int)(np.rint((s[0] )/step))
        #    ss = 0.0
        #    s_head  = np.full(head_number, 0.0)
        #    Is_head = np.full(head_number, 0.0)
        #    for i in range(head_number):
        #        s_head[i]  = ss
        #        Is_head[i] = np.exp(ss*ss*Rg*Rg/-3.0)
        #        ss += step
        #    s  = np.hstack((s_head, s))
        #    Is = np.hstack((Is_head, Is))

        # if len(Is[firstPointIndex:lastPointIndex]) != inputLength:
        #    print(f"{inputFilename} too short, skipping.")
        #    continue

        # if round(s[firstPointIndex], 3) != round(smin, 3):
        #    print(f"{inputFilename}: point {firstPointIndex} has s={s[firstPointIndex]}, expected s={smin}")
        #    exit()

        # if round(s[lastPointIndex - 1], 3) != round(smax, 3):
        #    print(f"{inputFilename}: point {lastPointIndex - 1} has s={s[lastPointIndex]}, expected s={smax}")
        #    exit()
        try:
            Is, __, __ = normalise(Is, stdIs, meanIs)
        except:
            pass
        test = np.array([Is, ])
        pred = loadedModel.predict(test)

        # TODO: instead of checking output number of points > 10 read model type (scalar/pddf)
        if len(pred[0]) > 10:  # pddf or autoencoder model
            # Find Dmax: first negative (or zero) point after max(p(r))
            max_pddf = np.argmax(pred)
            negIndex = np.argmax(pred[:, max_pddf:] <= 0)
            # Crop p(r > Dmax), nullify last point
            pred = pred[:, 0: (negIndex + max_pddf + 1)]
            pred[:, -1] = 0.0

            r = np.arange(0.0, len(pred[0]) * 0.25, 0.25)
            outCsv.append(inputFilename[:-4] + ', ' + str(round(r[-1], 3)))
            # print(f"{len(r)} - {len(pred[0])} - {r[-1]}") # DEBUG
            pddf_predicted = np.vstack((r, stdpddf * pred[0]))
            np.savetxt(inputFilename[:-4], np.transpose(pddf_predicted), fmt="%.8e")

        else:  # scalar model
            for number in pred[0]:
                outCsv.append(f"{inputFilename[:-4]},  {round(number, 3)}")

    if outCsvPath != "":
        np.savetxt(f"{outCsvPath}-{f}.csv", outCsv, delimiter=",", fmt='%s')
        print(f"{f}-{outCsvPath} is written.")
    else:
        print(f"Folder {f}:")
        print("\n".join(outCsv))
