#!/usr/bin/python
import argparse

parser = argparse.ArgumentParser(description='Compare NN predictions.')
parser.add_argument('csv1',  metavar='csv1',   type=str, help='path to the first csv file')
parser.add_argument('csv2',  metavar='csv2',   type=str, help='path to the second csv file')

parser.add_argument('-o', '--output', type=str, default="", help='save output in CSV format')

args = parser.parse_args()

import csv
import numpy as np
import os
import matplotlib.pyplot as plt
csv1 = args.csv1
csv2 = args.csv2

outCsvPath    = args.output

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

with open(outCsvPath, mode='w') as outfile:
    writer = csv.writer(outfile)
    for n in sameId:
        diff = round(float(dict1[n]) - float(dict2[n]), 2)
        writer.writerow([n, str(diff)])