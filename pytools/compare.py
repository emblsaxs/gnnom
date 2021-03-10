"""
Fast comparison of prediction against the ground truth.
Using CVS files and different metrics
"""
import argparse

parser = argparse.ArgumentParser(description='Compare NN predictions.')
parser.add_argument('csv1', metavar='csv1', type=str, help='path to the template csv file')
parser.add_argument('csv2', metavar='csv2', type=str, help='path to the csv file for comparison')

parser.add_argument('-o', '--output', type=str, default="",
                    help='save output in CSV format')
parser.add_argument('-m', '--metric', type=str, default="",
                    help='options: ad(absolute diff); rd(relative diff); h (histogram); l (csv1 vs csv2)')

args = parser.parse_args()

import csv
import numpy as np
import matplotlib.pyplot as plt

csv1 = args.csv1
csv2 = args.csv2

outCsvPath = args.output
metric = args.metric
ms = ['ad', 'rd', 'h', 'l']
if metric not in ms:
    print("wrong metric! dying...")
    exit()

# convert to dictionaries
with open(csv1, mode='r', newline='') as infile:
    reader = csv.reader(infile)
    dict1 = {rows[0]: rows[1] for rows in reader}

with open(csv2, mode='r', newline='') as infile:
    reader = csv.reader(infile)
    dict2 = {rows[0]: rows[1] for rows in reader}

# find intersections
fSet = set(dict1)
sSet = set(dict2)
sameId = []
for name in fSet.intersection(sSet):
    sameId.append(name)
print(f"{len(sameId)} out of {max(len(dict1), len(dict2))} ids are identical")

# write output csv
with open(outCsvPath, mode='w') as outfile:
    writer = csv.writer(outfile)
    ad = []  # absolute difference
    rd = []  # relative difference
    linGT = []  # ground truth
    linC = []  # predicted
    for num, n in enumerate(sameId, start=1):
        diff = round(float(dict2[n]) - float(dict1[n]), 4)  # absolute difference
        if metric == "l":
            linGT.append(round(float(dict1[n]), 4))
            linC.append(round(float(dict2[n]), 4))
            writer.writerow([round(float(dict1[n]), 4), round(float(dict2[n]), 4)])
        if metric == "ad":
            writer.writerow([str(num), str(diff), n])
            ad.append(diff)
        if metric in ["rd", "h"]:
            rDiff = diff / float(dict1[n])
            writer.writerow([str(num), str(rDiff), n])
            rd.append(rDiff)
outfile.close()

# plot absolute differences
if metric == "ad":
    plt.scatter(range(len(ad)), ad, c='tab:blue', alpha=0.3, edgecolors='none')
    plt.xlabel('Number of curve')
    plt.ylabel('Absolute difference')
    std = np.std(ad)
    med = np.median(ad)
    plt.title(f"std = {std * 100}%  median = {med * 100}%")
    plt.grid(True)
    plt.show()

# plot relative differences
if metric == "rd":
    plt.scatter(range(len(rd)), rd, c='tab:blue', alpha=0.3, edgecolors='none')
    plt.xlabel('Number of curve')
    plt.ylabel('Relative difference')
    rd = np.abs(rd)
    std = np.std(rd)
    med = np.median(rd)
    plt.title(f"std = {std * 100}%  median = {med * 100}%")
    plt.grid(True)
    plt.show()

# plot linear regression
if metric == "l":
    plt.scatter(linGT, linC, c='tab:blue', alpha=0.3, edgecolors='none')
    plt.xlabel('Ground truth')
    plt.ylabel('Predicted')
    linGT = np.array(linGT)
    linC = np.array(linC)
    rd = abs(linC - linGT) / linGT
    aver = "{:.2%}".format(np.mean(rd))
    med = "{:.2%}".format(np.median(rd))
    tt = f"Mean: {aver}\nMedian: {med}"
    plt.title(tt)
    plt.grid(True)
    plt.show()

# plot histogram
if metric == "h":
    # the histogram of the data
    n, bins, patches = plt.hist(rd, edgecolor='black', bins=50, density=True, facecolor='g', alpha=0.75)
    plt.xlabel('Bins')
    plt.ylabel('Probability')
    aver = "{:.2%}".format(np.mean(np.abs(rd)))
    med = "{:.2%}".format(np.median(np.abs(rd)))
    tt = f"Mean: {aver}\nMedian: {med}"
    plt.title(tt)
    plt.grid(True)
    plt.show()
