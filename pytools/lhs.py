#!/usr/bin/python
import argparse

parser = argparse.ArgumentParser(description='Uniformly selects distribution over two factors from a CSV file. '
                                             'Note: the file must contain id column')
parser.add_argument('csv', metavar='csv', type=str, help='path to the csv file')
parser.add_argument('f1', metavar='f1', type=str, help='first factor to distribute over')
parser.add_argument('f1Low', metavar='f1Low', type=str, help='low boundary for f1 factor')
parser.add_argument('f1High', metavar='f1High', type=str, help='higher boundary for f1 factor')
parser.add_argument('f2', metavar='f2', type=str, help='second factor to distribute over')
parser.add_argument('f2Low', metavar='f2Low', type=str, help='low boundary for f2 factor')
parser.add_argument('f2High', metavar='f2High', type=str, help='higher boundary for f2 factor')
parser.add_argument('num1', metavar='num1', type=str, help='number of bins')
parser.add_argument('num2', metavar='num2', type=str, help='number of proteins from each bin')

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
binsNum = int(args.num1)
numFromBin = int(args.num2)

print(f"Number of bins in each distribution is {binsNum}")
outputCsv = args.output
if os.path.exists(inputCsv):
    with open(inputCsv) as csvFile:
        csvReader = csv.DictReader(csvFile, delimiter=',')
        if f1 not in (csvReader.fieldnames):
            print(f"No such parameter {f1}!")
            exit()
        if f2 not in (csvReader.fieldnames):
            print(f"No such parameter {f2}!")
            exit()

        # to speed up the calculations let's make up a shorter array of dicts
        shortArr = []
        for row in csvReader:
            f1Float = float(row[f1])
            f2Float = float(row[f2])
            if f1Float >= f1Low and f1Float <= f1High:
                if f2Float >= f2Low and f2Float <= f2High:
                    shortArr.append({"id": row["id"], f1: f1Float, f2: f2Float})
else:
    print("No such file! ")
    exit()

# Do we need it?
f1Arr = [d[f1] for d in shortArr]
f2Arr = [d[f2] for d in shortArr]
f1Arr.sort()
f2Arr.sort()

finalList = []
# compute histograms
hist1, bins1 = np.histogram(f1Arr, bins=binsNum)
# pairs of left-right boundaries for each bin
pairs1 = zip(bins1, bins1[1:])
for i, bin in enumerate(pairs1):
    # array of proteins within that bin - clear every time
    proteinsFromBin = []
    # take binsNum from each bin -> total number will be binsNum**2 (WHY NOT??)
    for protein in shortArr:
        if (protein[f1] >= bin[0]) and (protein[f1] < bin[1]):
            proteinsFromBin.append(protein)

    # for all proteins within the bin build histogram and take 1 sample from each bin of the new distribution
    hist2, bins2 = np.histogram([d[f2] for d in proteinsFromBin], bins=binsNum)
    pairs2 = zip(bins2, bins2[1:])
    print(f"{i} bin has {len(proteinsFromBin)} proteins")
    for bin2 in pairs2:
        for j, protein2 in enumerate(proteinsFromBin):
            if (protein2[f2] >= bin2[0]) and (protein2[f2] < bin2[1]) and (j <= numFromBin):
                finalList.append(protein2)
                # DEBUG
                #print(f"bin1: {i}        bin2: {j}")

print(f"{len(finalList)} files is written to {outputCsv}")

# cdf = hist1.cumsum()
# cdf_normalized = cdf * hist1.max()/ cdf.max()
# plt.plot(cdf_normalized, color = 'b')

# write output csv
with open(outputCsv, mode='w') as outfile:
    writer = csv.writer(outfile, delimiter=',', quoting=csv.QUOTE_NONE)
    writer.writerow(['id'] + [f1] + [f2])
    for protein in finalList:
        # s = f"{protein['id']}, {protein[f1]}, {protein[f2]}"
        writer.writerow([protein['id']] + [protein[f1]] + [protein[f2]])

# Do we need it?
f3Arr = [d[f1] for d in finalList]
f4Arr = [d[f2] for d in finalList]

f3Arr.sort()
f4Arr.sort()

# plot histograms
fig, ax = plt.subplots(2, 2, tight_layout=True)
ax[0, 0].hist(f1Arr, edgecolor='black', bins=binsNum, facecolor='b', alpha=0.5)
ax[0, 0].set_xticks(np.linspace(min(f1Arr), max(f1Arr), 10))
ax[0, 0].legend([f"{f1} all"], loc="upper right")

ax[1, 0].hist(f2Arr, edgecolor='black', bins=binsNum, facecolor='g', alpha=0.5)
ax[1, 0].set_xticks(np.linspace(min(f2Arr), max(f2Arr), 10))
ax[1, 0].legend([f"{f2} all"], loc="upper right")

ax[0, 1].hist(f3Arr, edgecolor='black', bins=binsNum, facecolor='b', alpha=0.5)
ax[0, 1].set_xticks(np.linspace(min(f3Arr), max(f3Arr), 10))
ax[0, 1].legend([f"{f1} selected"], loc="upper right")

ax[1, 1].hist(f4Arr, edgecolor='black', bins=binsNum, facecolor='g', alpha=0.5)
ax[1, 1].set_xticks(np.linspace(min(f4Arr), max(f4Arr), 10))
ax[1, 1].legend([f"{f2} selected"], loc="upper right")

plt.show()
