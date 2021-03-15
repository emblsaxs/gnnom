"""
Fast comparison of prediction against the ground truth in batch mode.
Using CVS files
"""
import argparse

parser = argparse.ArgumentParser(description='Compare NN predictions.')
parser.add_argument('csv1', metavar='csv1', type=str, help='path to the template csv file')
parser.add_argument('csv2', metavar='csv2', type=str, help='path to the folder with csv files for comparison')

parser.add_argument('-o', '--output', type=str, default="",
                    help='save output in CSV format')

args = parser.parse_args()

import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import re

csv1 = args.csv1
folder = args.csv2

outCsvPath = args.output

# convert to dictionaries
with open(csv1, mode='r', newline='') as infile:
    reader = csv.reader(infile)
    dict1 = {rows[0]: rows[1] for rows in reader}

files = os.listdir(folder)
c = []  # concentrations parsed from the names of csv files
averRelError = []  # mean relative error
medianError = []  # median

# iterate over all csv files in the folder
for csv2 in files:
    result = re.search('dat-c(.*)-', csv2)
    wha = result.group(1)
    wha = wha.split("-")[0]
    if "025" in wha:
        conc = 0.25
    elif "05" in wha:
        conc = 0.5
    else:
        conc = float(wha)
    c.append(conc)
    # parse ids and predicted values
    path = os.path.join(folder, csv2)
    with open(path, mode='r', newline='') as infile:
        reader = csv.reader(infile)
        dict2 = {rows[0]: rows[1] for rows in reader}

    # find intersections
    fSet = set(dict1)
    sSet = set(dict2)
    sameId = []
    for name in fSet.intersection(sSet):
        sameId.append(name)
    print(f"{conc} mg/ml: {len(sameId)} out of {max(len(dict1), len(dict2))} ids are identical")

    # calculate discrepancies
    relDiff = []  # relative difference

    for num, n in enumerate(sameId, start=1):
        GT = float(dict1[n])
        P = float(dict2[n])
        AD = P - GT  # absolute difference
        RD = np.abs(AD) / GT
        relDiff.append(RD)

    aver = "{:.2%}".format(np.mean(relDiff))
    med = "{:.2%}".format(np.median(relDiff))
    averRelError.append(aver)
    medianError.append(med)

lists = sorted(zip(*[c, averRelError]))
x, y = list(zip(*lists))

out = []  # out csv to save or print out
for xx, yy in zip(x, y):
    out.append(f"{xx}, {yy}")

# plot
plt.scatter(x, y, facecolors='none', edgecolors='black')
plt.plot(x, y, c="black")
plt.gca().invert_yaxis()
plt.xlabel('Simulated concentration, mg/ml')
plt.ylabel('Relative error')
plt.grid(True)
plt.show()

# write output csv or print out
if outCsvPath != "":
    np.savetxt(outCsvPath, out, delimiter=",", fmt='%s')
    print(f"{outCsvPath} is written.")
else:
    print("\n".join(out))
