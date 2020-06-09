#!/usr/bin/env python
# coding: utf-8
import argparse

parser = argparse.ArgumentParser(description='Adds realistic errors to a synthetic data.')
parser.add_argument('dataPath',   metavar='path', type=str, help='path to the folder with synthetic data')
parser.add_argument('-t', '--teeth', type=str, help='path to the teeth')
parser.add_argument('-p', '--prefix', type=str, default="err_", help='prefix added to the output files')

args = parser.parse_args()


import numpy as np
import saxsdocument
import os


__, teeth = saxsdocument.read(args.teeth)

dataFiles = os.listdir(args.dataPath)
dataFiles.sort()

for inputFilename in dataFiles:
    try:
        if (os.path.isdir(os.path.join(args.dataPath, inputFilename))): continue 
        prop, cur  = saxsdocument.read(os.path.join(args.dataPath, inputFilename))
        s  = cur['s']
        sT = teeth['s']
        if (len(s) != len(sT)): 
            print(f"Warning: the length of {inputFilename} is different from {args.teeth}!")
            continue
        if(np.array_equal(s,sT) == False):
            print(f"Warning: grid in {inputFilename} is different from {args.teeth}!")
            print(str((s == sT).sum()) + " s-values out of " + str(len(s)) + " are equivalent")
        Is = cur['I']
        Err = cur['Err']
        IsT = teeth['I']
        ErrT = teeth['Err']
        #random_normal = np.random.normal(IT,ET)
        ErrI = np.multiply(IsT, np.sqrt(np.abs(Is)))
        Is = np.random.normal(Is, Err)
        #outCurve = np.vstack((s,Is, ErrI))
        prop['creator'] = 'addteeth'
        outCurve = {'s' : s, 'I' : Is, 'Err' : ErrI, 'Fit' : ''}
        saxsdocument.write(f"{args.prefix}{inputFilename}", outCurve, prop)

    except Exception as e:
        print(f"Error: Could not read {inputFilename}:")
        print(e)
        