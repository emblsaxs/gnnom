"""
Adds noise to the buffer and sample in .abs format, subtracts sample from buffer.
The most reliable method, but needs template SAXS data files.
The examples from EMBL p12 beamline are provided in the 'templates' directory
"""
import argparse

parser = argparse.ArgumentParser(
    description="Adds noise to the buffer and sample in .abs format."
                r"Example: python augment_with_buffer.py abs\ smoothed-water-final-x2-extrapolated.dat"
                r" template.dat -p tmp\ -ct 2 -a 0.00001 -b 0.0001 -c 0.00001")
parser.add_argument('dataPath', type=str, help='path to the folder with sample data')
parser.add_argument('-p', '--prefix', type=str, default="", help='add prefix to the rescaled files')
parser.add_argument('-ct', '--concentration', type=float, default=1.0, help='concentration of the sample to simulate')
parser.add_argument('buffer', type=str, help='smooth cell + buffer file on absolute scale')
parser.add_argument('noiseTemplate', type=str, help='the sample file on absolute scale for noise estimate')
# a,b,c -- coefficients for buffer augmentation
# sample = (sample + buffer) - c*(buffer + (a*s + b))
parser.add_argument('-a', '--slope', type=float, default=0,
                    help='Sigma for a. <a> = 0; sample = (sample + buffer) - c*(buffer + (A*s + b)')
parser.add_argument('-b', '--shift', type=float, default=0,
                    help='Sigma for b. <b> = 0; sample = (sample + buffer) - c*(buffer + (a*s + B)')
parser.add_argument('-c', '--scale', type=float, default=0,
                    help='Sigma for c. <c> = 1; sample = (sample + buffer) - C*(buffer + (a*s + b)')
parser.add_argument('--normalize-by-I0', type=bool, nargs='?', default=False, const=True,
                    help='Normalize by I(0) from the sample data')

args = parser.parse_args()

import os
import numpy as np
from gnnom.mysaxsdocument import saxsdocument

inputFolder = args.dataPath
prefix = args.prefix
conc = args.concentration
bufferPath = args.buffer
templatePath = args.noiseTemplate

aSigma = args.slope
bSigma = args.shift
cSigma = args.scale

is_norm = args.normalize_by_I0

# compute teeth from template file
tcurve, __ = saxsdocument.read(templatePath)
sI = np.array(tcurve['I'])
sErr = np.array(tcurve['Err'])
teeth = np.abs(sI / (sErr ** 2))

# create noise buffer
bcurve, __ = saxsdocument.read(bufferPath)
# sBuf         = np.array(bcurve['s'])
IsBuf = np.array(bcurve['I'])
# DEBUG
# errBuf       = np.sqrt(np.abs(IsBuf/teeth))
# IsBufNoise   = np.random.normal(IsBuf, errBuf)
# saxsdocument.write(f"{prefix}noise-{bufferPath}", {'s' : sBuf, 'I' : IsBufNoise, 'Err' : errBuf, 'Fit' : ''}, bprop)

# apply to all simulated data files on absolute scale
for inputFilename in os.listdir(inputFolder):
    filename, file_extension = os.path.splitext(inputFilename)
    if file_extension != '.abs': 
        print(f"Skipping {inputFilename}")
        continue
    dat, property = saxsdocument.read(os.path.join(inputFolder, inputFilename))
    property['sample-concentration'] = conc
    # create unsubtracted sample with noise
    s = np.array(dat['s'])
    I0 = dat['I'][0]
    Is = conc * np.array(dat['I']) + IsBuf
    err = np.sqrt(np.abs(Is / teeth))
    IsNoise = np.random.normal(Is, err)

    # create augmented buffer with noise
    a = np.random.normal(0, aSigma)
    b = np.random.normal(0, bSigma)
    c = np.random.normal(1, cSigma)
    # augment buffer
    IsBufAug = c * IsBuf + (a * s + b)
    # errBufAug   = errBuf * c
    errBufAug = np.sqrt(np.abs(IsBufAug / teeth))
    # add noise
    IsBufAugNoise = np.random.normal(IsBufAug, errBufAug)
    # subtract one from the other
    IsSub = (IsNoise - IsBufAugNoise) / conc
    errSub = np.sqrt(errBufAug ** 2 + err ** 2) / conc

    # normalize by I(0) from the smooth sample data
    if (is_norm):
        IsSub = IsSub / I0
        errSub = errSub / I0
    # save file
    saxsdocument.write(f"{os.path.join(prefix, inputFilename[:-4])}.dat", {'s': s, 'I': IsSub, 'Err': errSub}, property)
