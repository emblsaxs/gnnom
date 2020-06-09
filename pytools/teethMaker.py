#!/usr/bin/env python
# coding: utf-8

import numpy as np
import saxsdocument
import os
import sys
import glob
import time

start_time = time.time()
def main(outFileName, *args):
    Is = np.array([])
    s = np.array([])
    errs = np.array([])
    teeth = np.array([])
    lenS = []
    i = 0
    for file in args:
        if outFileName in file:
            continue
        doc = saxsdocument.read(file)
        dat = np.transpose(np.array(doc.curve[0]))
        i+=1
        if i == 1:
            s = dat[0]
            Is = dat[1]
            errs = dat[2]
            teeth = np.divide(errs,np.sqrt(Is))
            lenS.append(len(s))
        else:
            scat_vector = dat[0]
            intensity = dat[1]
            error = dat[2]
            tooth = np.divide(error,np.sqrt(intensity))
            s = np.vstack((s,scat_vector))
            Is = np.vstack((Is,intensity))
            errs = np.vstack((errs,error))
            teeth = np.vstack((teeth, tooth))
            lenS.append(len(scat_vector))

    np.info(teeth)
    teethN = np.transpose(teeth)
    sPoints= min(lenS)
    len_n =Is.shape[0]

    mean_teeth = np.zeros(sPoints)
    stdv_teeth = np.zeros(sPoints)
    i=0
    for n in range(0, sPoints):
        av_t = np.mean(teethN[n])
        st_t = np.std(teethN[n])
        mean_teeth[i]=av_t
        stdv_teeth[i]=st_t
        i+=1

    mean_tooth_dat = np.vstack((s[0],mean_teeth, stdv_teeth))
    np.savetxt(outFileName, np.transpose(mean_tooth_dat), fmt = "%.8e")

    print ("Averaged file is written to ... "+outFileName)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "-h":
        print("example: python teethMaker.py data/*.dat -o=myTeeth.dat")
        sys.exit("Bye")
    if len(sys.argv) < 2:
        raise SyntaxError("Insufficient arguments")
    else:
        args = iter(sys.argv)
        isArgument = False
        for arg in args:
            if "=" in arg:
                a = arg.split("=")[0]
                n = arg.split("=")[1]
                if (a == "-o"):
                    main(n, *sys.argv[2:])
                    isArgument = True
        if (isArgument == False):
            main("teeth.dat", *sys.argv[2:])
