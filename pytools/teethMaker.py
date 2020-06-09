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
        Is  = cur['I']
        #if (len(Is) != 100): continue
        s = cur['s']
        Err = cur['Err']
        teeth = np.divide(np.array(Err), np.sqrt(np.abs(Is)))
        outTeeth.append(teeth)

    except Exception as e:
        print(f"Error: Could not read {inputFilename}:")
        print(e)


outTeeth = np.array(outTeeth)
averageIs = np.mean(outTeeth, axis = 0)
stdvIs = np.std(outTeeth, axis = 0)
prop = {"creator" : "teethMaker"}
#out = np.vstack((s, averageIs, stdvIs))
out = {'s' : s, 'I' : averageIs, 'Err' : stdvIs, 'Fit' : ''}

saxsdocument.write(args.output, out, prop)
print("--- %s seconds ---" % (time.time() - start_time))
