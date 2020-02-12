#!/usr/bin/python
import argparse
import saxsdocument
import numpy as np
import os

parser = argparse.ArgumentParser(description='Randomly scales SAXS *.dat files within the given standard deviation.')
parser.add_argument('dataPath',   metavar='path', type=str, help='path to the folder with data')
parser.add_argument('-p', '--prefix', type=str, default="", help='add prefix to the rescaled files')
parser.add_argument('-n', '--noise', type=float,  default =0.0, help='sigma of Gaussian noise (standard deviation)')
parser.add_argument('-b', '--background', type=float,  default =0.0, help='sigma of Gaussian background constant (standard deviation)')
parser.add_argument('-s', '--scale', type=float,  default =0.0, help='sigma of Gaussian scale factor (standard deviation)')


args = parser.parse_args()

inputFolder   = args.dataPath
prefix        = args.prefix
noise         = args.noise
bckgrd        = args.background
scale         = args.scale

for inputFilename in os.listdir(inputFolder):
    try:
        mul = np.random.normal(1.0, scale, 1)
        add = np.random.normal(0.0, bckgrd, 1)
        doc   = saxsdocument.read(os.path.join(inputFolder, inputFilename))
        dat   = np.transpose(np.array(doc.curve[0]))
        s    = dat[0]
        Is   = dat[1]
        err  = dat[2]
        nrd = np.random.normal(0.0, noise, len(Is))
        errm = np.sqrt((err*mul)**2 + (noise*mul)**2)
        Ism  = (Is + nrd + add) * mul
        out  = np.vstack((s,Ism, errm))
        outPath = (prefix + inputFilename)
        np.savetxt( outPath, np.transpose(out), fmt = "%.8e")
        print(outPath + ': is written to the disk...')
    except Exception as e:
        print(e)
