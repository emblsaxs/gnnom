#!/usr/bin/python
import argparse
import saxsdocument
import numpy as np
import os

parser = argparse.ArgumentParser(description='Randomly scales SAXS *.dat files within the given standard deviation.')
parser.add_argument('dataPath',   metavar='path', type=str, help='path to the folder with data')
parser.add_argument('-p', '--prefix', type=str, default="", help='add prefix to the rescaled files')
parser.add_argument('-s', '--stdev', type=float,  default =0.01, help='sigma (standard deviation)')


args = parser.parse_args()

inputFolder   = args.dataPath
prefix        = args.prefix
stdev         = args.stdev

for inputFilename in os.listdir(inputFolder):
    try:
        scale = np.random.normal(1.0, stdev, 1)
        doc   = saxsdocument.read(os.path.join(inputFolder, inputFilename))
        dat   = np.transpose(np.array(doc.curve[0]))
        s    = dat[0]
        Is   = dat[1]
        err  = dat[2]
        errm = err*scale
        Ism  = Is * scale
        out  = np.vstack((s,Ism, errm))
        outPath = (prefix + inputFilename)
        np.savetxt( outPath, np.transpose(out), fmt = "%.8e")
        print(outPath + ': is written to the disk...')
    except:
        print("Error: Could not read input data")
