#!/usr/bin/env python
# coding: utf-8

import numpy as np
import saxsdocument
import os
import sys
import glob
import time

start_time = time.time()
def main(prefix, teethFile, *args):
    docT = saxsdocument.read(teethFile)
    for file in args:
        if "-p=" in file or "-t=" in file:
            continue
        docI = saxsdocument.read(file)
        datT = np.transpose(np.array(docT.curve[0]))
        sT = datT[0]
        IT = datT[1]
        ET = datT[2]
        datI = np.transpose(np.array(docI.curve[0]))
        sI = datI[0]
        II = datI[1]
        if(np.array_equal(sT,sI) == False):
            print("Error in " + file + "! Grids in dat and teeth files are different")
            print(str((sI == sT).sum()) + " s-values out of " + str(len(sI)) + " are equivalent")
        #random_normal = np.random.normal(IT,ET)
        EI = np.multiply(IT, np.sqrt(II))
        II = np.random.normal(II, EI)
        PCA_with_errors = np.vstack((sI,II, EI))
        outFile = str(prefix) + os.path.basename(file)
        np.savetxt(outFile, np.transpose(PCA_with_errors), fmt = "%.8e")
        print ("New file is created ... " + outFile)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "-h":
        print("example: python addteeth.py -t=myTeeth.dat data-reduced/*.dat -p=data-reduced-with-errors/err_")
        sys.exit("Bye")
        raise SyntaxError("Insufficient arguments")
    if len(sys.argv) < 3:
        raise SyntaxError("Insufficient arguments")
    else:
        args = iter(sys.argv)
        prefix = "err"
        for arg in args:
            if "-p" in arg:
                prefix = arg.split("=")[1]
            if "-t" in arg:
                teethFile = arg.split("=")[1]

        if (teethFile != ""):
            main(prefix, teethFile, *sys.argv[2:])

        else:
            raise SyntaxError("Parse error")
