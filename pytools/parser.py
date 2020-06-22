#!/usr/bin/env
import argparse

parser = argparse.ArgumentParser(description='Parse crysol log files and plot histo.')
parser.add_argument('logPath',     metavar='logs',   type=str, help='path to the training log folder')
parser.add_argument('parameter',   metavar='par', type=str, help='mw/dmax/rg')
parser.add_argument('outCsv', type=str,   default="out.csv", help='path to output csv file')

args = parser.parse_args()
import os
import numpy as np
import matplotlib.pyplot as plt
logs = args.logPath
par   = args.parameter
fileNames = os.listdir(logs)
print("Parsing log files...")
outCsv     = []
params     = []

for file in fileNames:
    f = os.path.join(logs,file)
    with open(f, 'r') as logFile:
      lines = logFile.readlines()
      for line in lines:
              if par == "rg":
                  if "slope" in line:
                      rg = float(line.split()[-1])
                      params.append(rg)
                      outCsv.append(file[:-4] + ', ' + str(round(rg, 3)))
                      break
              if par == "dmax":
                  if "diameter" in line:
                      dmax = float(line.split()[-1])
                      params.append(dmax)
                      outCsv.append(file[:-4] + ', ' + str(round(dmax, 3)))
                      break         
              if par == "mw":
                  if "Weight" in line:
                      mw = float(line.split()[2])/1000.0
                      params.append(mw)
                      outCsv.append(file[:-4] + ', ' + str(round(mw, 3)))
                      break
print("...done.")

#save ground true values to csv
outCsvPath = f"ground-{par}-{len(fileNames)}.csv"
np.savetxt(outCsvPath, outCsv, delimiter=",", fmt='%s')
print(outCsvPath + " is written.")

n, bins, patches = plt.hist(params, edgecolor='black', bins=50, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Bins')
plt.ylabel('Number')
aver = round(np.mean(params), 2)
minim = round(min(params), 2)
maxim = round(max(params), 2)   
tt = f"{par}\nMean: {aver}\nMin: {minim}  Max: {maxim}"
plt.title(tt)
plt.grid(True)
plt.show()
