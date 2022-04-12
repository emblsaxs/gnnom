"""
Selects subset of the most distant data sets using any metric.
Can be used to select the most representative training set for a NN.
"""

import argparse

parser = argparse.ArgumentParser(description='Selects subset of the most distant data sets')
parser.add_argument('dataPath', metavar='path', type=str, help='path to the folder with data')
parser.add_argument('percent', metavar='percent', type=str, help='percent of data to choose')
parser.add_argument('outPath', metavar='out', type=str, help='path to the output folder')
parser.add_argument('-p', '--prefix', type=str, default="", help='add prefix to the output files')

args = parser.parse_args()

import itertools
import os
import random
import numpy as np
from gnnom.mysaxsdocument import saxsdocument


def distance(point1, point2):
    """Find a distance using any kind of metric"""
    point1 = np.array(point1)
    point2 = np.array(point2)
    d = 10 - np.dot(point1, point2)  # use dot product of vectors
    # DEBUG
    # d = np.linalg.norm(point1 - point2)
    # d = 0 # do not convert list to numpy - no advantage in speed
    # for i in range(len(point1)):
    #    d += pow((point1[i]-point2[i]),2)
    return d


def maxDistancePointAndList(point, list):
    """Finds the maximum distance between one (N-dimensional) point and an an array of
    such points. Returns {point, distance}"""
    dist = 0
    for p in list:
        # FIXME: any metric may be used here
        d = distance(point, p)
        if d > dist:
            dist = d
            distP = p
    return {"Point": distP, "Distance": dist}


def distancePointAndList(point, list):
    dist = 0
    for p in list:
        dist += distance(p, point)
    return dist


inputFolder = args.dataPath
prefix = args.prefix
percent = float(args.percent) / 100.0
outputFolder = args.outPath

inFiles = os.listdir(inputFolder)
# find the number of curves to create:
numberPoints = (int)(len(inFiles) * percent)
print(f"Found: {len(inFiles)} Expected output: {numberPoints}")
# create a matrix of the second columns
inMatrix = []
inCurves = []

# generate a giant matrix
for inputFilename in inFiles:
    try:
        prop, curve = saxsdocument.read(os.path.join(inputFolder, inputFilename))
        r = curve['s']
        p = curve['I']
        inMatrix.append(p[::4])
        inCurves.append({"filename": inputFilename, "properties": prop, "data": p})
    except Exception as e:
        print(e)

# remove duplicates from inMatrix
length = len(inMatrix)
inMatrix.sort()
inMatrix = list(k for k, _ in itertools.groupby(inMatrix))
if (length - len(inMatrix) > 0): print(f"{length - len(inMatrix)} duplicates removed.")

# pick a random curve to be the first point
firstCurve = random.choice(inMatrix)
inMatrix.remove(firstCurve)
outMatrix = []
outMatrix.append(firstCurve)
for point in range(numberPoints - 1):
    print(f"{point} out of {numberPoints - 1}...")
    # find the most distance point in respect to the found ones
    dist = 0
    for p in inMatrix:
        d = distancePointAndList(p, outMatrix)
        if dist < d:
            dist = d
            pp = p
    inMatrix.remove(pp)
    outMatrix.append(pp)
    # save files
    r = np.arange(101)
    curve = [i for i in inCurves if (i["data"][::4] == pp)][0]
    name = curve["filename"]
    prop = curve["properties"]
    pddf = curve["data"]
    path = os.path.join(outputFolder, prefix, name)
    saxsdocument.write(f"{prefix}{path}", {'s': r, 'I': pddf, 'Err': '', 'Fit': ''}, prop)

    print(f"{name} is saved")
