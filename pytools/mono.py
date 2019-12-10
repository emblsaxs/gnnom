#!/usr/bin/python
import argparse
import saxsdocument
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


parser = argparse.ArgumentParser(description='Finds monodisperse regions across the chromotographic peaks.')
parser.add_argument('dataPath',   metavar='path', type=str, help='path to the folder with data')

args = parser.parse_args()

inputFolder   = args.dataPath
fileList      = os.listdir(inputFolder)
fileList.sort()
Ilist         = []
allLength     = len(fileList)
i             = 0
for inputFilename in fileList:
    try:
        i += 1
        doc   = saxsdocument.read(os.path.join(inputFolder, inputFilename))
        dat   = np.transpose(np.array(doc.curve[0]))
        s     = dat[0]
        Is    = dat[1]
        Ilist.append(Is)
        if i%10 ==0:
            print(str(i) + " out of " + str(allLength) + " are downloaded")
    except:
        print("Error: Could not read input data " + inputFilename)
print("All files are downloaded...")
Imat = np.array((Ilist))
index1 = np.min(np.where(s > 0.1))
index2 = np.max(np.where(s < 0.8))
chrom = (Imat[:, index1:index2])
chrom = np.mean(chrom, axis = 1)
# we need only one major peak
peaks = []
i = 0
while len(peaks) != 1:
    i += 1
    threshold = max(chrom) - i*(max(chrom) + min(chrom))/10.0
    peaks, _ = find_peaks(chrom, distance = 50, height=threshold)
# let's find the monodisperse region around this peak
rng = [5, 25, 50, 100, 200]
peaks  = int(peaks)
i = 0
for r in rng:
    i += 1
    x      = np.arange(peaks - r, peaks + r)
    forsvd = Imat[:, peaks - r: peaks + r]
    u, s, vh = (np.linalg.svd(forsvd))
    #print("For the range of " + str(2*r) + " frames the first 5 singular values: ")
    #print(s[0:5])

    plt.subplot(len(rng),1,int(i))
    s = "sigma-1 = " + str(round(s[0],1)) + " sigma-2 = " + str(round(s[1],1)) + " sigma-3 = " + str(round(s[2],1))
    plt.plot(chrom, label = s)
    plt.legend(loc = "upper right")
    plt.xlabel('Number of frame')
    plt.ylabel('Average intensity')
    plt.plot(x, chrom[x], "x")
    plt.hlines(threshold, 0, len(chrom), color="gray")

plt.show()


