"""
A simple script to parse parameters from CRYSOL log files and show a heat map.
Can be used to estimate how diverse is your training set.
"""
import argparse

parser = argparse.ArgumentParser(description='Parse crysol log files and plot histo.')
parser.add_argument('logPath', metavar='logs', type=str, help='path to the training log folder')
parser.add_argument('parameter', metavar='par', type=str, help='mw/dmax/rg')
parser.add_argument('--secondParameter', '-par2', default=None, type=str, help='mw/dmax/rg/vshell/vexc')

args = parser.parse_args()
import os
import numpy as np
import matplotlib.pyplot as plt
from gnnom.utils.crysollog import parseCrysolLogs

logs = []
path = args.logPath
par = args.parameter
for file in os.listdir(path):
    logs.append(os.path.join(path, file))
params, outCsv = parseCrysolLogs(logs, par)
pars = [item for sublist in params for item in sublist]
# if we want eg mw vs rg
if args.secondParameter:
    from scipy.stats import gaussian_kde

    par2 = args.secondParameter
    params2, __ = parseCrysolLogs(logs, par2)
    pars2 = [item for sublist in params2 for item in sublist]
    # plt.scatter(pars, pars2, facecolors='none', edgecolors='black')
    xs = np.array(pars)
    ys = np.array(pars2)
    xy = np.vstack([xs, ys])
    zs = gaussian_kde(xy)(xy)
    idx = zs.argsort()
    xs, ys, zs = xs[idx], ys[idx], zs[idx]
    fig, ax = plt.subplots()
    ax.set_facecolor('xkcd:black')
    ax.scatter(xs, ys, c=zs)
    # ax.scatter(xs, ys, c=zs, s=10, edgecolor='')

    ax.set(xlabel=par, ylabel=par2)
    plt.grid(True)
    plt.show()
else:
    # save ground true values to csv
    outCsvPath = f"ground-{par}-{len(logs)}.csv"
    np.savetxt(outCsvPath, outCsv, delimiter=",", fmt='%s')
    print(outCsvPath + " is written.")
    n, bins, patches = plt.hist(pars, edgecolor='black', bins=50, density=True, facecolor='g', alpha=0.75)
    plt.xlabel('Bins')
    plt.ylabel('Number')
    aver = round(np.mean(pars), 2)
    minim = round(np.min(pars), 2)
    maxim = round(np.max(pars), 2)
    tt = f"{par}\nMean: {aver}\nMin: {minim}  Max: {maxim}"
    plt.title(tt)
    plt.grid(True)
    plt.show()
