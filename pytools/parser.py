#!/usr/bin/env

import re,sys,os
import numpy as np
import fnmatch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# python hist_rg (input_root_log_directory) (rg/dmax/mw) (output_csv)

if (sys.argv[1] == '-h') or (sys.argv[1] == '--help'):
  print("python parser (input_root_log_directory) (rg/dmax/mw) (output_csv)")
  sys.exit()

folderPath = sys.argv[1]
par        = sys.argv[2]
csvPath  = sys.argv[3]

fileNames = []
params    = []

for root, dirnames, filenames in os.walk(folderPath):
  for filename in fnmatch.filter(filenames, '*.log'):
    f = os.path.join(root,filename)
    fileNames.append(filename)
    with open(f, 'r') as logFile:
      lines = logFile.readlines()
      for line in lines:
        if par == 'rg':
          if re.search('Rg from the slope of net intensity',line):
            Rg = line.split(':')[-1]
            params.append(Rg)
        elif par == 'dmax':
          if re.search('Envelope  diameter',line):
            Dmax = line.split(':')[-1]
            params.append(Dmax)
        elif par == 'mw':
          if re.search('Molecular Weight',line):
            l1 = line.split('Molecular Weight: ')[1]
            l2 = l1.split('   Dry volume         :')[0]
            params.append(l2)
        if line == None:
          print('Parse error for file: ' + f)

nfileNames = np.array(fileNames)
nParams    = np.array(params)
out        = np.vstack((nfileNames, nParams))
out        = out.astype(str)
#np.info(out)
np.savetxt(csvPath,np.transpose(out), delimiter = ',', fmt = "%s")

nParams    = nParams.astype(float)

plt.hist(nParams, bins = 'auto')

plt.ylabel('Number of files')
plt.xlabel(par)

plt.title('Histogram of ' + par)

plt.savefig('hist_'+par+'.png',format='png', dpi=250)
plt.clf()
