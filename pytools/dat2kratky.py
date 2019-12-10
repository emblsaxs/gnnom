#!/usr/bin/python
import argparse
import saxsdocument
import numpy as np
import os

parser = argparse.ArgumentParser(description='Converts SAXS *.dat files to their Kratky representation.')
parser.add_argument('dataPath',   metavar='path', type=str, help='path to the folder with data')
parser.add_argument('-p', '--prefix', type=str, default="", help='add prefix to the Kratky files')

args = parser.parse_args()

inputFolder   = args.dataPath
prefix        = args.prefix

for inputFilename in os.listdir(inputFolder):
    try:
        doc  = saxsdocument.read(os.path.join(inputFolder, inputFilename))
        dat  = np.transpose(np.array(doc.curve[0]))
        s  = dat[0]
        Is = dat[1]
        Ik = s*s*Is
        out = np.vstack((s,Ik))
        outPath = (prefix + inputFilename)
        np.savetxt( outPath, np.transpose(out), fmt = "%.8e")
        print(outPath + ': is written to the disk...')
    except:
        print("Error: Could not read input data")
