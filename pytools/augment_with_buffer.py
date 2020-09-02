#!/usr/bin/python
import argparse
import pysaxsdocument as saxsdocument
import numpy as np
import os

parser = argparse.ArgumentParser(description='Adds noise to the buffer and sample in .abs format. Example: python pytools\augment_with_buffer.py abs\ smoothed-water-final-x2_rebin.dat sec_0015_01220_stretched_rebinned_norm.dat -p tmp\ -ct 2 -a 0.00001 -b 0.0001 -c 0.00001')
parser.add_argument('dataPath',   metavar='path', type=str, help='path to the folder with sample data')
parser.add_argument('-p', '--prefix', type=str, default="", help='add prefix to the rescaled files')
parser.add_argument('-ct', '--concentration', type=float,  default =1.0, help='concentration of the sample to simulate')
parser.add_argument('buffer', type=str, help='smooth cell + buffer file on absolute scale')
parser.add_argument('noiseTemplate', type=str, help='the sample file on absolute scale for noise estimate')
# a,b,c -- coefficients for buffer augmentaion
# sample = (sample + buffer) - c*(buffer + (a*s + b))
parser.add_argument('-a', '--slope',type=float, default = 0, help='Sigma for a. <a> = 0; sample = (sample + buffer) - c*(buffer + (A*s + b)')
parser.add_argument('-b', '--shift',type=float, default = 0,help='Sigma for b. <b> = 0; sample = (sample + buffer) - c*(buffer + (a*s + B)')
parser.add_argument('-c', '--scale',type=float, default = 0,help= 'Sigma for c. <c> = 1; sample = (sample + buffer) - C*(buffer + (a*s + b)')


args = parser.parse_args()

inputFolder   = args.dataPath
prefix        = args.prefix
conc          = args.concentration
bufferPath    = args.buffer
templatePath  = args.noiseTemplate

aSigma        = args.slope
bSigma        = args.shift
cSigma        = args.scale

# compute teeth from template file
tcurve   = saxsdocument.read(templatePath)
sI           = np.array(tcurve.curve[1])
sErr         = np.array(tcurve.curve[2])
teeth        = np.abs(sI/(sErr**2))

# create noise buffer
bcurve   = saxsdocument.read(bufferPath)
#sBuf         = np.array(bcurve['s'])
IsBuf        = np.array(bcurve.curve[1])
#errBuf       = np.sqrt(np.abs(IsBuf/teeth))
#IsBufNoise   = np.random.normal(IsBuf, errBuf)
#saxsdocument.write(f"{prefix}noise-{bufferPath}", {'s' : sBuf, 'I' : IsBufNoise, 'Err' : errBuf, 'Fit' : ''}, bprop)

# apply to all simulated data files on absolute scale
for inputFilename in os.listdir(inputFolder):

    dat   = saxsdocument.read(os.path.join(inputFolder, inputFilename))
    # create unsubtracted sample with noise
    s    = np.array(dat.curve[0])
    Is   = conc*np.array(dat.curve[1]) + IsBuf
    err  = np.sqrt(np.abs(Is/teeth))
    IsNoise = np.random.normal(Is, err)
    
    # create augmented buffer with noise
    a    = np.random.normal(0, aSigma)
    b    = np.random.normal(0, bSigma)
    c    = np.random.normal(1, cSigma)
    # augment buffer
    IsBufAug    = c*IsBuf + (a*s + b)
    #errBufAug   = errBuf * c
    errBufAug    = np.sqrt(np.abs(IsBufAug/teeth))
    # add noise
    IsBufAugNoise   = np.random.normal(IsBufAug, errBufAug)
    # subtract one from the other
    IsSub  = (IsNoise - IsBufAugNoise)/conc
    errSub = np.sqrt(errBufAug**2 + err**2)/conc
    # save file
    saxsdocument.write(f"{prefix}{inputFilename[:-4]}.dat", [s, IsSub, errSub], dat.property)

