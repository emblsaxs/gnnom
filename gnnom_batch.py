#!/usr/bin/python
import argparse

parser = argparse.ArgumentParser(description='Apply NN model in batch regime.')
parser.add_argument('architecture',  metavar='json',   type=str, help='path to the json file with architecture')
parser.add_argument('weights', metavar='h5',   type=str, help='path to the hdf5 file with weights')
parser.add_argument('dataPath',   metavar='path', type=str, help='path to the folder with data')
parser.add_argument('--first', type=int, default=1,  help='index of the first point to use (default: 1)')
parser.add_argument('--last',  type=int, default=-1, help='index of the last point to use (default: use all)')
parser.add_argument('-o', '--output', type=str, default="", help='save output in CSV format')

args = parser.parse_args()

from keras.models import model_from_json
import saxsdocument
import numpy as np
import os

jsonFilename  = args.architecture
h5Filename    = args.weights
inputFolder   = args.dataPath
outCsvPath    = args.output
firstPointIndex = int(args.first) - 1


outputLength  = 302 # p(r) points

try:
    # load json and create model
    jsonFile = open(jsonFilename, 'r')
    loadedModelJson = jsonFile.read()
    jsonFile.close()
    loadedModel = model_from_json(loadedModelJson)
    # load weights into new model
    loadedModel.load_weights(h5Filename)
    inputLength = loadedModel.input_shape[1]  # I(s) points

    if firstPointIndex >= inputLength:
        print("--first should be less than " + str(inputLength))
        exit()

    if int(args.last) == -1:
        lastPointIndex = inputLength
    elif int(args.last) > inputLength:
        print("--last should be less than " + str(inputLength))
        exit()
    else:
        lastPointIndex = int(args.last)
    print("Model loaded. Yeah!")
except Exception as e:
    print("Error: Oops, model can not be uploaded.")
    print(e)
    exit()

# output csv
outCsv = []
for inputFilename in os.listdir(inputFolder):
    try:
        doc  = saxsdocument.read(os.path.join(inputFolder, inputFilename))
        dat  = np.transpose(np.array(doc.curve[0]))
        s  = dat[0][firstPointIndex:lastPointIndex]
        Is = dat[1][firstPointIndex:lastPointIndex]
        if len(Is) != lastPointIndex - firstPointIndex:
            continue
    except:
        print("Error: Could not read input data")


    # Assume Smin is correct; fill up to required Smax with typical I(0.4) intensity (assuming I(0) = 1.0)
    #if(len(Is) > inputLength): print("Too many points in the data: " + str(len(Is)) + "points")
    #zeroes = np.full((inputLength - len(Is)), Is[-1])
    #IsExtended = np.concatenate((Is, zeroes))


    # evaluate loaded model on test data
    #loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    #score = loaded_model.evaluate(np.array(Is, np.array(PDDF[0:n_cases]),verbose=0)
    #print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    r = np.arange(0.0, outputLength, 1.0)

    #test = np.array([IsExtended, ])
    test = np.array([Is, ])
    pred = loadedModel.predict(test)

    #TODO: multiply by p(r) normalization coefficient
    if len(pred[0]) == outputLength:
        point = len(pred[0])
        r = r[:point]
        pred = pred[:,:point]
        pred[:,-1] = 0
        pddfPredicted = np.vstack((r, pred))
        np.savetxt('pddf-predicted.dat', np.transpose(pddfPredicted), fmt = "%.8e")
        print('pddf-predicted.dat is written to the disk...')

    elif len(pred[0]) < 10:
        for number in pred[0]:
            outCsv.append(inputFilename + ', ' + str(round(number, 3)))

if outCsvPath != "":
    np.savetxt(outCsvPath, outCsv, delimiter=",", fmt='%s')
    print(outCsvPath + " is written.")
else:
    print ("\n".join(outCsv))
