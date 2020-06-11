#!/usr/bin/python
import argparse
import saxsdocument
import numpy as np
import os

parser = argparse.ArgumentParser(description='Adds noise to the buffer and sample in .abs format. Example: python pytools\augment.py  int-with-noise-norm-i0\ -p int-with-noise-x2-norm-i0\ -n 0.001 -b 0.001 -s 0.5')
parser.add_argument('dataPath',   metavar='path', type=str, help='path to the folder with sample data')
parser.add_argument('-p', '--prefix', type=str, default="", help='add prefix to the rescaled files')
parser.add_argument('-c', '--concentration', type=float,  default =1.0, help='concentration of the sample to simulate')
parser.add_argument('buffer', type=str, help='smooth cell + buffer file on absolute scale')
parser.add_argument('noiseTemplate', type=str, help='the sample file on absolute scale for noise estimate')


args = parser.parse_args()

inputFolder   = args.dataPath
prefix        = args.prefix
conc          = args.concentration
bufferPath    = args.buffer
templatePath  = args.noiseTemplate

# compute teeth from template file
__, tcurve   = saxsdocument.read(templatePath)
sI           = np.array(tcurve['I'])
sErr         = np.array(tcurve['Err'])
teeth        = np.abs(sI/(sErr**2))

# create noise buffer
bprop, bcurve   = saxsdocument.read(bufferPath)
sBuf         = np.array(bcurve['s'])
IsBuf        = np.array(bcurve['I'])
errBuf       = np.sqrt(np.abs(IsBuf/teeth))
IsBufNoise   = np.random.normal(IsBuf, errBuf)
saxsdocument.write(f"{prefix}noise-{bufferPath}", {'s' : sBuf, 'I' : IsBufNoise, 'Err' : errBuf, 'Fit' : ''}, bprop)

# apply to all simulated data files on absolute scale
for inputFilename in os.listdir(inputFolder):
    try:
        prop, curve   = saxsdocument.read(os.path.join(inputFolder, inputFilename))
        # create unsubtracted sample with noise
        s    = np.array(curve['s'])
        Is   = conc*np.array(curve['I']) + IsBuf
        err  = np.sqrt(np.abs(Is/teeth))
        IsNoise = np.random.normal(Is, err)
        #prop['teeth'] = templatePath
        # unsubtracted curve
        #saxsdocument.write(f"{prefix}{inputFilename}.dat", {'s' : s, 'I' : IsNoise, 'Err' : err, 'Fit' : ''}, prop)
        IsSub  = (IsNoise - IsBufNoise)/conc
        errSub = np.sqrt(errBuf**2 + err**2)/conc
        saxsdocument.write(f"{prefix}{inputFilename[:-4]}.dat", {'s' : s, 'I' : IsSub, 'Err' : errSub, 'Fit' : ''}, prop)
    except Exception as e:
        print(e)
