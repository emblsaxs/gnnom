"""
Fast comparison of prediction against the ground truth for multiple folders.
Use CVS files of type 'dat-cXXX.csv' with ground truth values, where XXX stands for concentration.
"""
import argparse

parser = argparse.ArgumentParser(description='Compare NN predictions.')
parser.add_argument('csv1', metavar='csv1', type=str, help='path to the template csv file')
parser.add_argument('csv2', metavar='csv2', type=str, help='path to the folder with csv files for comparison')
parser.add_argument('--col1', type=int, default=1, help='Column number from the template csv file (starts from 0)')
parser.add_argument('--col2', type=int, default=1,
                    help='Column number from the csv files for comparison (starts from 0)')
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


# helper function
def representsPositiveFloat(s):
    try:
        f = float(s)
        if f >= 0:
            return True
        else:
            return False
    except ValueError:
        return False


# convert to dictionaries
with open(csv1, mode='r', newline='') as infile:
    reader = csv.reader(infile)
    dict1 = {rows[0]: rows[args.col1] for rows in reader}

files = os.listdir(folder)
c = []  # concentrations parsed from the names of csv files
averRelError = []  # mean relative error
medianError = []  # median
standDev = []  # standard deviation
outliers = []  # outliers

# iterate over all csv files in the folder
for csv2 in files:
    if not csv2.endswith(".csv"): continue
    if 'dat-c' not in csv2:
        print(f"Skipping csv file {csv2}")
        continue
    result = re.search('dat-c(.+).csv', csv2)
    wha = result.group(1)
    if "025" in wha:
        conc = 0.25
    elif "05" in wha:
        conc = 0.5
    else:
        conc = float(wha)
    c.append(conc)
    # parse ids and predicted values
    path = os.path.join(folder, csv2)
    dict2 = {}
    outlierNum = 0
    with open(path, mode='r', newline='') as infile:
        reader = csv.reader(infile)
        for num, rows in enumerate(reader):
            if (len(rows) <= args.col2) or not representsPositiveFloat(rows[args.col2]):
                print(f"Can't parse {rows} Line: {num} File: {csv2}")
                outlierNum += 1
                continue
            else:
                dict2[rows[0]] = float(rows[args.col2]) / 1.000

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
        try:
            GT = float(dict1[n])
            P = float(dict2[n])
            AD = P - GT  # absolute difference
            RD = np.abs(AD) / GT
            relDiff.append(RD)
        except Exception as e:
            print(e)

    aver = "{:.2%}".format(np.mean(relDiff))
    med = "{:.2%}".format(np.median(relDiff))
    std = "{:.2%}".format(np.std(relDiff))
    averRelError.append(aver)
    medianError.append(med)
    standDev.append(std)
    outliers.append(outlierNum)

lists = sorted(zip(*[c, averRelError, medianError, standDev, outliers]))
x, y, z, sd, on = list(zip(*lists))

out = []  # out csv to save or print out
out.append(f"Concentration, average relative error, median, standard deviation, outliers")
for xx, yy, zz, ss, onon in zip(x, y, z, sd, on):
    out.append(f"{xx}, {yy}, {zz}, {ss}, {onon}")

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
