"""
Selects subset of the most distant data sets from the median.
Can be used to select the most representative training set for a NN.
Legacy code.
"""

import argparse
import os
import random
from collections import namedtuple

import numpy as np

from gnnom.mysaxsdocument import saxsdocument

parser = argparse.ArgumentParser(description='Selects subset of the most distant data sets')
parser.add_argument('dataPath', metavar='path', type=str, help='path to the folder with data')
parser.add_argument('a', type=int, help='number of median neighbours to select')
parser.add_argument('b', type=int, help='number of median neighbours to keep')
parser.add_argument('outPath', metavar='out', type=str, help='path to the output folder')
parser.add_argument('-p', '--prefix', type=str, default="", help='add prefix to the output files')

args = parser.parse_args()

inputFolder = args.dataPath
prefix = args.prefix
outputFolder = args.outPath

inFiles = os.listdir(inputFolder)
inFiles.sort()

# create a matrix of the second columns

inMatrix = []

for inputFilename in inFiles:
    try:
        prop, curve = saxsdocument.read(os.path.join(inputFolder, inputFilename))
        p = curve['I']
        inMatrix.append(p)
    except Exception as e:
        print(e)

inMatrix = np.array(inMatrix)
median = np.median(inMatrix, axis=0)
distances = np.linalg.norm(inMatrix - median, axis=1)

# r = np.arange(101)
# path = os.path.join(outputFolder, prefix, "sub01.dat")
# saxsdocument.write(f"{prefix}{path}", {'s' : r, 'I' : xx[2], 'Err' : '', 'Fit' : ''}, prop)
# exit()

distance = namedtuple('distance', 'filename distance')
distancesTuple = []

for i in range(len(distances)):
    distancesTuple.append(distance(inFiles[i], distances[i]))

distancesTuple.sort(key=lambda x: getattr(x, 'distance'))

medianNeighbours = distancesTuple[:args.a]
random.shuffle(medianNeighbours)

s = []
for x in distancesTuple[args.a:]:
    s.append(x[0])
for x in medianNeighbours[:args.b]:
    s.append(x[0])
print(" ".join(s))
