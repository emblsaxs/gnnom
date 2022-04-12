"""
Randomly scales SAXS *.dat files within the given standard deviation
"""
import argparse

parser = argparse.ArgumentParser(
    description="Randomly scales SAXS *.dat files within the given standard deviation."
                r" Example: python augment.py  int-with-noise\ -p int-norm-i0\ -n 0.001 -b 0.001 -s 0.5")
parser.add_argument('dataPath', metavar='path', type=str, help='path to the folder with data')
parser.add_argument('-p', '--prefix', type=str, default="", help='add prefix to the rescaled files')
parser.add_argument('-n', '--noise', type=float, default=0.0, help='sigma of Gaussian noise (standard deviation)')
parser.add_argument('-b', '--background', type=float, default=0.0,
                    help='sigma of Gaussian background constant (standard deviation)')
parser.add_argument('-s', '--scale', type=float, default=0.0,
                    help='sigma of Gaussian scale factor (standard deviation)')

args = parser.parse_args()

import os
import numpy as np
from gnnom.mysaxsdocument import saxsdocument

inputFolder = args.dataPath
prefix = args.prefix
noise = args.noise
bckgrd = args.background
scale = args.scale

# run for all files in directory
for inputFilename in os.listdir(inputFolder):
    try:
        mul = np.random.normal(1.0, scale)
        add = np.random.normal(0.0, bckgrd)
        __, curve = saxsdocument.read(os.path.join(inputFolder, inputFilename))
        s = np.array(curve['s'])
        Is = np.array(curve['I'])
        err = np.array(curve['Err'])
        if len(err) == 0: err = np.zeros(len(Is))  # np.sqrt(Is)
        nrd = np.random.normal(0.0, noise, len(Is))
        nrd[-1] = 0
        errm = np.sqrt((err * mul) ** 2 + (noise * mul) ** 2)
        Ism = (Is + nrd + add) * mul
        out = np.vstack((s, Ism, errm))
        outPath = os.path.join(prefix, inputFilename)
        np.savetxt(outPath, np.transpose(out), footer=f"noise: {noise}\nbackground: {add}\nscale:{mul}", fmt="%.8e")
    except Exception as e:
        print(f"Error with file {inputFilename}: {e}")
