"""
Fast comparison of prediction against the ground truth.
Using CVS files and different metrics
"""
import argparse

parser = argparse.ArgumentParser(description='Compare NN predictions.')
parser.add_argument('csv1', metavar='csv1', type=str, help='path to the template csv file')
parser.add_argument('csv2', metavar='csv2', type=str, help='path to the csv file for comparison')
parser.add_argument('--col1', type=int, default=1, help='Column number from the template csv file (starts from 0)')
parser.add_argument('--col2', type=int, default=1,
                    help='Column number from the csv file for comparison (starts from 0)')

parser.add_argument('-o', '--output', type=str, default="",
                    help='save output in CSV format')
parser.add_argument('-m', '--metric', type=str, default="l",
                    help='options: ad(absolute diff); rd(relative diff); h (histogram); l (csv1 vs csv2)')

args = parser.parse_args()

import os.path
import csv
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"

csv1 = args.csv1
csv2 = args.csv2

outCsvPath = args.output
metric = args.metric
ms = ['ad', 'rd', 'h', 'l']
if metric not in ms:
    print("Wrong metric! dying...")
    exit()

# convert to dictionaries
with open(csv1, mode='r', newline='') as infile:
    reader = csv.reader(infile)
    dict1 = {rows[0]: rows[args.col1] for rows in reader}

with open(csv2, mode='r', newline='') as infile:
    reader = csv.reader(infile)
    dict2 = {rows[0]: rows[args.col2] for rows in reader}

# find intersections
fSet = set(dict1)
sSet = set(dict2)
sameId = []
for name in fSet.intersection(sSet):
    sameId.append(name)
print(f"{len(sameId)} out of {max(len(dict1), len(dict2))} ids are identical")

# calculate discrepancies
absDiff = []  # absolute difference
relDiff = []  # relative difference
groundTruth = []  # ground truth
predicted = []  # predicted
out = []  # csv to save or to print out
outliers = 0
for id in sameId:
    GT = float(dict1[id])  # * 1000  # sasflow results: to convert from Da to kDa
    P = float(dict2[id])
    AD = P - GT  # absolute difference
    RD = np.abs(AD) / GT
    if P > 0:
        absDiff.append(AD)
        relDiff.append(RD)
    else:
        outliers += 1
    groundTruth.append(GT)
    predicted.append(P)
    if metric == "ad":
        out.append(f"{id}, {round(AD, 3)}")
    else:
        out.append(f"{id}, {round(RD, 3)}")
# compute mean error and median
if metric == "ad":
    aver = "{:.2%}".format(np.mean(absDiff))
    med = "{:.2%}".format(np.median(absDiff))
else:
    aver = "{:.2%}".format(np.mean(relDiff))
    med = "{:.2%}".format(np.median(relDiff))

# plot absolute differences
if metric == "ad":
    plt.scatter(range(len(absDiff)), absDiff, c='tab:blue', alpha=0.3, edgecolors='none')
    plt.xlabel('Number of curve')
    plt.ylabel('Absolute difference')

# plot relative differences
if metric == "rd":
    plt.scatter(range(len(relDiff)), relDiff, c='tab:blue', alpha=0.3, edgecolors='none')
    plt.xlabel('Number of curve')
    plt.ylabel('Relative difference')

# plot linear regression
if metric == "l":
    # plt.scatter(groundTruth, predicted, c='tab:blue', alpha=0.3, edgecolors='none')
    # plt.xlabel('Ground truth')
    # plt.ylabel('Predicted')
    linex = [0, max(groundTruth)]
    liney = [0, max(groundTruth)]
    predictedFilename = os.path.splitext(os.path.basename(csv2))[0]
    fig = px.scatter(x=groundTruth, y=predicted, hover_name=sameId,
                     title=f"{predictedFilename}: "
                           f"{aver} - average relative error, {med} - median relative error. "
                           f"Number of negative outliers: {outliers}")
    fig.add_shape(type="line",
                  x0=0,
                  y0=0,
                  x1=max(groundTruth),
                  y1=max(groundTruth))
    fig.update_layout(xaxis_title="Ground truth (x)", yaxis_title="Predicted (y)")
    fig.show()
    print(f"{aver} - average relative error")
    print(f"{med} - median relative error")
    print(f"{outliers} outlier(s) omitted")

# plot histogram
if metric == "h":
    # the histogram of the data
    n, bins, patches = plt.hist(relDiff, edgecolor='black', bins=50, density=True, facecolor='g', alpha=0.75)
    plt.xlabel('Bins')
    plt.ylabel('Number per bin')

# plt.title(f"Mean: {aver}\nMedian: {med}")
# plt.grid(True)
# plt.show()

# write output csv or print out
if outCsvPath != "":
    np.savetxt(outCsvPath, out, delimiter=",", fmt='%s')
    print(f"{outCsvPath} is written.")
else:
    print("\n".join(out))
