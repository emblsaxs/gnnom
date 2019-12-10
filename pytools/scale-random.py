#!/usr/bin/python
import argparse
import saxsdocument
import numpy as np
import os

parser = argparse.ArgumentParser(description='Randomly scales SAXS *.dat files uniformly within the given range.')
parser.add_argument('dataPath',   metavar='path', type=str, help='path to the folder with data')
parser.add_argument('-p', '--prefix', type=str, default="", help='add prefix to the rescaled files')
parser.add_argument('-d', '--down', type=float,  default =0, help='low boundary for rescaling')
parser.add_argument('-u', '--upper', type=float, default = 1, help='high boundary for rescaling')


args = parser.parse_args()

inputFolder   = args.dataPath
prefix        = args.prefix
low           = args.down
high          = args.upper

for inputFilename in os.listdir(inputFolder):
    try:
        scale = np.random.uniform(low, high, 1)
        doc   = saxsdocument.read(os.path.join(inputFolder, inputFilename))
        dat   = np.transpose(np.array(doc.curve[0]))
        s  = dat[0]
        Is = dat[1]
        Ism = Is * scale
        out = np.vstack((s,Ism))
        outPath = (prefix + inputFilename)
        np.savetxt( outPath, np.transpose(out), fmt = "%.8e")
        print(outPath + ': is written to the disk...')
    except:
        print("Error: Could not read input data")
