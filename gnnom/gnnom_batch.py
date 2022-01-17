# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to apply NN for prediction.
"""
folders = ["dat-c025", "dat-c05", "dat-c1", "dat-c2", "dat-c4", "dat-c8", "dat-c16"]
# folders = ["dat-c16"]
import argparse

parser = argparse.ArgumentParser(description='Apply NN model in batch regime.')
parser.add_argument('architecture', metavar='json', type=str, help='path to the json file with architecture')
parser.add_argument('weights', metavar='h5', type=str, help='path to the hdf5 file with weights')
parser.add_argument('dataPath', metavar='path', type=str, help='path to the folder with data')
parser.add_argument('-o', '--output', type=str, default="", help='prefix to output CSV files')

args = parser.parse_args()

from keras.models import model_from_json
from mysaxsdocument import saxsdocument
import numpy as np
import os
import json
from normalisation.meanvariance import normalise
import time

time_begin = time.time()

jsonFilename = args.architecture
h5Filename = args.weights
inputFolder = args.dataPath
outCsvPath = args.output
multiplier = 1.0

try:
    # load json and create model
    jsonFile = open(jsonFilename, 'r')
    loadedModelJson = jsonFile.read()
    json_data = json.loads(loadedModelJson)
    # Optional fields in json
    if 'Normalization coefficient' in json_data:
        multiplier = float(json_data['Normalization coefficient'])
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
    inputLength = lastPointIndex - firstPointIndex

    jsonFile.close()
    loadedModel = model_from_json(loadedModelJson)
    # load weights into new model
    loadedModel.load_weights(h5Filename)
    inputLength = loadedModel.input_shape[1]  # I(s) points
    print(f"Expected input: {inputLength} points.")
    # outputLength = loadedModel.output_shape[1]  # p(r) points
    print("Model loaded. Yeah!")
    print(f"Time passed: {time.time() - time_begin}")

except KeyError as e:
    print(f"Error: Oops, model cannot be loaded! Missing value: {e}")
    quit()

except Exception as e:
    print(f"Error: {e}")
    quit()


dataFiles = os.listdir(args.dataPath)
dataFiles.sort()

for f in folders:
    inputIs = []
    inputBasenames = []
    outCsv = []
    print(f"Processing folder: {f}")
    t = os.path.join(args.dataPath, f)
    dataFiles = os.listdir(t)
    dataFiles.sort()
    for inputFilename in dataFiles:
        try:
            cur, __ = saxsdocument.read(os.path.join(t, inputFilename))

            s = np.round(cur['s'], 3)
            firstSIndex = np.where(s == round(smin, 3))[0][0]
            lastSIndex = np.where(s == round(smax, 3))[0][0] + 1

            if (lastSIndex - firstSIndex) != inputLength:
                print(f"{inputFilename} wrong grid, skipping.")
                continue
            Is = cur['I'][firstSIndex:lastSIndex]
            Err = cur['Err'][firstSIndex:lastSIndex]

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
        # for i in range(100):
        try:
            # Iss = np.random.normal(Is, Err)
            Iss, __, __ = normalise(Is, stdIs, meanIs)
            inputIs.append(Iss)
            inputBasenames.append(inputFilename[:-4])  #+str(i))
        except:
            inputIs.append(Is)
            pass

    test = np.array(inputIs)
    time_start = time.time()
    pred = loadedModel.predict_on_batch(test)
    pred = pred.flatten()
    print(f"{time.time() - time_start} - apply model")
    outCsv = np.transpose([inputBasenames, np.round(multiplier * pred, 3)])

    if outCsvPath != "":
        np.savetxt(f"{outCsvPath}-{f}.csv", outCsv, delimiter=",", fmt='%s')
        print(f"{outCsvPath}-{f}.csv is written.")
    else:
        print(f"Folder {f}:")
        print("\n".join(outCsv))

print(f"All done. Time passed: {time.time() - time_begin}")
