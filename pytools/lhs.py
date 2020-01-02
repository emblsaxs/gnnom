#!/usr/bin/python
import argparse

parser = argparse.ArgumentParser(description='Uniformly selects distribution over two factors from a CSV file.')
parser.add_argument('csv', metavar='csv', type=str, help='path to the csv file')
parser.add_argument('f1', metavar='f1', type=str, help='first factor to distribute over')
parser.add_argument('f1Low', metavar='f1Low', type=str, help='low boundary for f1 factor')
parser.add_argument('f1High', metavar='f1High', type=str, help='higher boundary for f1 factor')
parser.add_argument('f2', metavar='f2', type=str, help='second factor to distribute over')
parser.add_argument('f2Low', metavar='f2Low', type=str, help='low boundary for f2 factor')
parser.add_argument('f2High', metavar='f2High', type=str, help='higher boundary for f2 factor')
parser.add_argument('num', metavar='num', type=str, help='number of entries to select')

parser.add_argument('-o', '--output', type=str, default="", help='save output in CSV format')

args = parser.parse_args()

import csv
import numpy as np
import os
import matplotlib.pyplot as plt

inputCsv = args.csv
f1 = args.f1
f1Low = float(args.f1Low)
f1High = float(args.f1High)
f2 = args.f2
f2Low = float(args.f2Low)
f2High = float(args.f2High)
num = args.num
binsNum = (int)(np.sqrt(float(num)))
print(f"number of bins in each distribution is {binsNum}")
out = args.output
if os.path.exists(inputCsv):
    with open(inputCsv) as csvFile:
        csvReader = csv.DictReader(csvFile, delimiter=',')
        if f1 not in (csvReader.fieldnames):
            print(f"No such parameter {f1}!")
            exit()
        if f2 not in (csvReader.fieldnames):
            print(f"No such parameter {f2}!")
            exit()
        # save the arrays
        f1Arr = []
        f2Arr = []
        for row in csvReader:
            if float(row[f1]) >= f1Low and float(row[f1]) <= f1High:
                f1Arr.append(float(row[f1]))
            if float(row[f2]) >= f2Low and float(row[f2]) <= f2High:
                f2Arr.append(float(row[f2]))

f1Arr.sort()
f2Arr.sort()

# built histograms
# cdf = hist1.cumsum()
# cdf_normalized = cdf * hist1.max()/ cdf.max()

# plt.plot(cdf_normalized, color = 'b')

fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)

ax1.hist(f1Arr, edgecolor='black', bins=binsNum, facecolor='b', alpha=0.5, label=f1)
ax2.hist(f2Arr, edgecolor='black', bins=binsNum, facecolor='g', alpha=0.5, label=f2)

ax1.set_xticks(np.linspace(min(f1Arr), max(f1Arr), 10))
ax1.legend([f1], loc="upper right")
ax2.legend([f2], loc="upper right")
ax2.set_xticks(np.linspace(min(f2Arr), max(f2Arr), 10))
# print (np.linspace(f1Low, f1High, binsNum))
plt.show()
