#!/usr/bin/python
import argparse

parser = argparse.ArgumentParser(description='Compare NN predictions.')
parser.add_argument('csv1',  metavar='csv1',   type=str, help='path to the template csv file')
parser.add_argument('csv2',  metavar='csv2',   type=str, help='path to the csv file for comparison')

parser.add_argument('-o', '--output', type=str, default="", help='save output in CSV format')
parser.add_argument('-m', '--metric', type=str, default="", help='options: d(diff); sd(squared diff); h (histo)')

args = parser.parse_args()

import csv
import numpy as np
import os
import matplotlib.pyplot as plt
csv1 = args.csv1
csv2 = args.csv2

outCsvPath    = args.output
metric        = args.metric
ms = ['d', 'sd', 'h']
if metric not in ms:
    print("wrong metric! exiting...")
    os._exit(0)
# convert to dictionaries
with open(csv1, mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
       dict1 = {rows[0]:rows[1] for rows in reader}

with open(csv2, mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
       dict2 = {rows[0]:rows[1] for rows in reader}

# find intersecions
fSet   = set(dict1)
sSet   = set(dict2)

sameId   = []
for name in fSet.intersection(sSet):
    sameId.append(name)
print(str(len(sameId)) + " out of " + str(max(len(dict1), len(dict2))) + " ids are identical")
with open(outCsvPath, mode='w') as outfile:
    writer = csv.writer(outfile)
    histo = []
    for n in sameId:
        diff = round(float(dict1[n]) - float(dict2[n]), 2)
        if metric == "d":
            writer.writerow([n, str(diff)])
        if metric == "sd":
            writer.writerow([n, str(round(diff**2/float(dict1[n]),2))])
        if metric == "h":
            histo.append(diff)
outfile.close()

if metric == "h":
    # the histogram of the data
    n, bins, patches = plt.hist(histo, 50, density=True, facecolor='g', alpha=0.75)

    plt.xlabel('Bins')
    plt.ylabel('Probability')
    std = np.std(diff)
    med = np.median(diff)
    plt.title("std = " + str(std) + "\nmedian = " + str(med))
    plt.grid(True)
    plt.show()