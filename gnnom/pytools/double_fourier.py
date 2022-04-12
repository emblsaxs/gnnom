"""
Double Fourier transform to validate the predicted P(r) functions.
Not applicable to scalar models (legacy code).
"""

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from gnnom.mysaxsdocument import saxsdocument

pi = math.pi


def dir_ff(s, Is, Err, name):
    smin = np.min(s)
    smax = np.max(s)
    if smin >= smax: print("Smin > Smax!")
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.clear()
    ax1.set_xlabel("Inverse Angstrom")
    ax1.set_ylabel("Intensity")
    ax1.semilogy(s, Is, color='b')
    ax1.axvline(x=smin, color='r')
    ax1.axvline(x=smax, color='r')
    markers, caps, bars = ax1.errorbar(s, Is, yerr=Err, capsize=1,
                                       elinewidth=1, markeredgewidth=1, ecolor='grey', color='b')
    [bar.set_alpha(0.5) for bar in bars]
    [cap.set_alpha(0.5) for cap in caps]

    r_max = 200
    fourier_pred = np.zeros(r_max * 10)
    r = np.arange(0.0, r_max, 0.1)
    for i, rr in enumerate(r):
        hit = s * s * np.multiply(Is, np.sinc(s * rr / pi))
        fourier_pred[i] = (rr * rr / 2 * pi * pi) * np.trapz(np.transpose(hit)) * 0.000004
        if fourier_pred[i] < 0 and i > 10:
            fourier_pred[i:] = 0
            break
    ax2.clear()
    ax2.set_xlabel("Angstrom")
    ax2.set_ylabel("PDDF")
    ax2.axhline(y=0, color='g')
    ax2.plot(r, fourier_pred, color='b')
    # four_pddf = np.vstack(r, fourier_pred)
    # np.savetxt("dima_fourier.dat", np.transpose(four_pddf), fmt = "%.8e")
    delta_s = s[1] - s[0]
    s_double = np.arange(0.0, np.max(s), delta_s)
    double_fourier = np.zeros(len(s_double))
    for i, ss in enumerate(s_double):
        tih = np.multiply(fourier_pred, np.sinc(ss * r / pi))
        double_fourier[i] = 4 * pi * np.trapz(np.transpose(tih))
    ax1.plot(s_double, double_fourier, '--', color='r')
    fig.canvas.draw()
    # compute a sum of P(r) second derivatives
    sd = np.abs(np.sum(np.diff(fourier_pred, 2)))
    plt.savefig(name + '.png', format='png', dpi=250)
    # Save the fit files 
    ff = np.vstack((s_double, double_fourier))
    np.savetxt(name + ".fit.dat", np.transpose(ff), fmt="%.6e")
    # Sew the head
    Is_merged = 0
    for i, ss in enumerate(s_double):
        if ss < smin: Is_merged = np.hstack((Is_merged, double_fourier[i]))
        if ss > smin:
            Is_merged = np.hstack((Is_merged, Is))
            break
    s_merged = np.linspace(start=0.0, stop=smax, num=len(Is_merged))
    f_merg = np.vstack((s_merged, Is_merged))
    np.savetxt(name + "-merged.dat", np.transpose(f_merg), fmt="%.6e")


def main():
    path = sys.argv[1]
    # path = "jupyter_scripts/big_data-top-1023/3aqz_pdb1.dat"
    # path = "SASDGB6.dat"
    doc = saxsdocument.read(path)
    dat = np.transpose(np.array(doc.curve[0]))
    s = np.array(dat[0])
    Is = np.array(dat[1])
    Err = np.array(dat[2])
    dir_ff(s, Is, Err, os.path.basename(path))


if __name__ == '__main__':
    main()
