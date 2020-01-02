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
num = float(args.num)
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
        # to speed up the calculations let's make up a shorter array of dicts {id : [f1, f2]}
        shortArr = []
        for row in csvReader:
            if float(row[f1]) >= f1Low and float(row[f1]) <= f1High:
                if float(row[f2]) >= f2Low and float(row[f2]) <= f2High:
                    f1Arr.append(float(row[f1]))
                    f2Arr.append(float(row[f2]))
                    shortArr.append({row["id"]: [float(row[f1]), float(row[f2])]})
else:
    print("No such file you dumbass! ")
    exit()

f1Arr.sort()
f2Arr.sort()

finalList = []
# compute histograms
hist1, bins1 = np.histogram(f1Arr, bins=binsNum)
pairs1 = zip(bins1, bins1[1:])
for k, bin in enumerate(pairs1):
    # array of second factor values
    f2FromBin = []
    f2FromBinArr = []
    # take (int) num/len(pairs) the most different over f2 elements
    numFromEachBin = (num / len(bins1))
    for protein in shortArr:
        f1Val = list(protein.values())[0][0]
        # for all proteins within the bin build histogram and take numFromEachBin from each bin
        if (f1Val >= bin[0]) and (f1Val < bin[1]) and (k <= numFromEachBin):
            f2Val = list(protein.values())[0][1]
            f2FromBin.append({list(protein.keys())[0]: f2Val})
            f2FromBinArr.append(f2Val)
    hist2, bins2 = np.histogram(f2FromBinArr, bins=binsNum)
    pairs2 = zip(bins2, bins2[1:])
    print(len(f2FromBin))
    for i, bin2 in enumerate(pairs2):
        for protein2 in f2FromBin:
            f2Val = list(protein2.values())[0]
            if (f2Val >= bin2[0]) and (f2Val < bin2[1]) and (i <= numFromEachBin):
                finalList.append(protein2)

print(len(finalList))

# cdf = hist1.cumsum()
# cdf_normalized = cdf * hist1.max()/ cdf.max()
# plt.plot(cdf_normalized, color = 'b')
# plot histograms
fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)

ax1.hist(f1Arr, edgecolor='black', bins=binsNum, facecolor='b', alpha=0.5, label=f1)
ax2.hist(f2Arr, edgecolor='black', bins=binsNum, facecolor='g', alpha=0.5, label=f2)

ax1.set_xticks(np.linspace(min(f1Arr), max(f1Arr), 10))
ax1.legend([f1], loc="upper right")
ax2.legend([f2], loc="upper right")
ax2.set_xticks(np.linspace(min(f2Arr), max(f2Arr), 10))

plt.show()
