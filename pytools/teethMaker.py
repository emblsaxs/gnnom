#!/usr/bin/python
import argparse

parser = argparse.ArgumentParser(description='Creates a file that can be used for simulation of errors.')
parser.add_argument('dataPath',   metavar='path', type=str, help='path to the folder with experimental data')
parser.add_argument('-o', '--output', type=str, default="out.dat", help='save output file')

args = parser.parse_args()

import numpy as np
import saxsdocument
import os
import time

start_time = time.time()

lenS = []
outTeeth = []
dataFiles = os.listdir(args.dataPath)
dataFiles.sort()

for inputFilename in dataFiles:
    try:
        __, cur  = saxsdocument.read(os.path.join(args.dataPath, inputFilename))
        s  = cur['s']
        Is = cur['I']
        Err = cur['Err']
        teeth = np.divide(Err, np.sqrt(Is))
        outTeeth.append(teeth)

    except Exception as e:
        print(f"Error: Could not read {inputFilename}:")
        print(e)


outTeeth = np.array(outTeeth)
averageIs = np.mean(outTeeth, axis = 0)
stdvIs = np.std(outTeeth, axis = 0)

saxsdocument.write(args.output, np.vstack((s, averageIs, stdvIs)), {creator : "teethMaker"})
print ("Averaged file is written to ... "+ args.output)
print("--- %s seconds ---" % (time.time() - start_time))
