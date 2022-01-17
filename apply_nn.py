"""
Apply NN for prediction MW and Dmax.
Data are first resampled to get estimation of uncertainties
"""
import argparse

parser = argparse.ArgumentParser(description='Apply NN model.')
parser.add_argument('type', type=str, help='p (protein), idp (intrinsically disordered protein) or na'
                                           '(nucleic acid)')
parser.add_argument('parameter', type=str, help='mw (molecular weight) or dmax (maximum intraparticle distance)')
parser.add_argument('dataPath', metavar='path', type=str, help='path to the data file')
parser.add_argument('I0', type=float, help='intensity in origin from AUTORG')
parser.add_argument('Rg', type=float, help='radius of gyration from AUTORRG')
parser.add_argument('--units', type=str, default='nanometer', help='angular units: angstrom or nanometer')
parser.add_argument('--n', default=1000, type=int, help='how many times to resample')
# parser.add_argument('-o', '--output', type=str, default="", help='prefix to output CSV files')

args = parser.parse_args()
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import model_from_json
from gnnom.mysaxsdocument import saxsdocument
import numpy as np
import json
import sys
# from normalisation.meanvariance import normalise
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

# import time
set_matplotlib_formats('svg')
# from scipy import stats

smax3 = 1.0
smax2 = 4.980390e-01
smax1 = 1.960780e-01
smin0 = 0.0196078
multiplier = 1
# check arguments
mType = args.type
if mType not in ['p', 'idp', 'na']:
    parser.error("Wrong type of molecule! Please choose between p, idp and na.")
par = args.parameter
if par not in ['mw', 'dmax']:
    parser.error("Wrong Parameter! Please choose between mw and dmax.")
units = args.units
if units not in ['angstrom', 'nanometer']:
    parser.error("Wrong units! Please choose between ANGSTROM and NANOMETER.")
n = args.n
inputFilename = args.dataPath
I0 = args.I0
Rg = args.Rg


def normalise(Is, divisor=None, subtractor=None):
    """Normalise data as x - <x> / Var(x)"""
    Is = np.array(Is)
    if subtractor is None: subtractor = np.mean(Is, axis=0)
    Is = Is - subtractor
    if divisor is None: divisor = np.var(Is, axis=0)
    Is = np.divide(Is, divisor)
    where_are_NaNs = np.isnan(Is)
    Is[where_are_NaNs] = 0.0
    where_are_Infs = np.isinf(Is)
    Is[where_are_Infs] = 0.0
    return Is, divisor, subtractor


# read saxs data, find smin and smax
try:
    cur, __ = saxsdocument.read(inputFilename)
    s = cur['s']
    if units == "nanometer":
        s = [ss / 10.0 for ss in s]  # change to angstroms
        Rg = Rg / 10.0
    smin = min(s)
    smax = max(s)
    if smin > smin0:
        print(f"Error: Insufficient angular range! smin = {smin} > {smin0} A^-1")
        sys.exit(0)
    if smax >= smax3:
        lastIndex = 256
    elif smax >= smax2:
        lastIndex = 129
    elif smax >= smax1:
        lastIndex = 52
    else:
        print(f"Insufficient angular range!"
              f"smax = {smax} < {smax1} A^-1")
        sys.exit(0)
    I = np.divide(cur['I'], I0)
    Err = np.divide(cur['Err'], I0)

except Exception as e:
    print(f"Error: Could not read {inputFilename}:")
    raise Exception(e)

# read appropriate model
try:
    modelPath = os.path.join("models", f"smax-index-{lastIndex}", f"{par}-3l-80u-{mType}",
                             f"gnnom-{par}-5-{lastIndex}-e100-u80")
    jsonFilename = modelPath + ".json"
    # load json and create model
    jsonFile = open(jsonFilename, 'r')
    loadedModelJson = jsonFile.read()
    json_data = json.loads(loadedModelJson)
    # Optional fields in json
    mw_kda = 1.0
    if 'Normalization coefficient' in json_data:
        multiplier = float(json_data['Normalization coefficient'])
    if 'meanIs' in json_data:
        meanIs = json_data['meanIs']
        stdIs = json_data['stdIs']
    elif 'meanIs' not in json_data:
        # print(f"WARNING! "
        #      f"{jsonFilename} does not contain normalization coefficients!"
        #      f"Proceeding without normalization...")
        mw_kda = 0.001
    # Compulsory fields in json
    smin = float(json_data['smin'])
    smax = float(json_data['smax'])
    firstPointIndex = int(json_data['firstPointIndex'])
    lastPointIndex = int(json_data['lastPointIndex'])

    jsonFile.close()
    loadedModel = model_from_json(loadedModelJson)
    # load weights into new model
    h5Filename = modelPath + ".h5"
    loadedModel.load_weights(h5Filename)
    inputLength = loadedModel.input_shape[1]  # I(s) points
    # print(f"Expected input: {inputLength} points.")
    # outputLength = loadedModel.output_shape[1]  # p(r) points
    # print("Model loaded. Yeah!")

except KeyError as e:
    raise Exception(f"Error: Oops, model cannot be loaded! Missing value: {e}")

except Exception as e:
    raise Exception(f"Error: {e}")

# generate a grid for model
sModel = np.linspace(smin, smax, inputLength)
halfStep = (sModel[1] - sModel[0]) / 2

# regrid data file to the grid from model
sNew, INew, ErrNew = ([] for i in range(3))
for sm in sModel:
    sTemp, ITemp, ErrTemp = ([] for i in range(3))
    for se, ie, erre in zip(s, I, Err):
        if se - halfStep < sm <= se + halfStep:
            sTemp.append(se)
            ITemp.append(ie)
            ErrTemp.append(erre)
        elif sm > smax + halfStep:
            break  # to speed up
    sNew.append(np.mean(sTemp))
    INew.append(np.mean(ITemp))
    er = np.sqrt(sum(np.square(ErrTemp))) / len(ErrTemp)
    ErrNew.append(er)
# # DEBUG
# plt.scatter(s, np.log10(I), c='blue', alpha=0.5, edgecolors='black')
# plt.plot(sNew, np.log10(INew), c='red')
# plt.show()
# saxsdocument.write("SASDH39-regrid3.dat", {'s': sNew, 'I': INew, 'Err': ErrNew})
# start = time.time()
# resample n times and run nn to do prediiction
p = []
data = []
for i in range(n):
    Is = np.random.normal(INew, ErrNew)
    # saxsdocument.write(f"{inputFilename}-resample-{i}.dat", {"s": sModel, "I": Is, 'Err': ErrNew})
    # exit()
    try:
        Is, __, __ = normalise(Is, stdIs, meanIs)
        data.append(Is)
    except:
        data.append(Is)
        pass

data = np.array(data)
p = loadedModel.predict_on_batch(data)
p = multiplier * p * mw_kda
if par == 'mw':
    p = p[p > 0]
elif par == 'dmax':
    p = p[p > 2 * Rg]
# end = time.time()
# print(f"{par} mean : {round(np.mean(p), 2)}\n"
#      f"{par} std  : {round(np.std(p), 2)}")
# print(f"TOTAL TIME: {end - start}")
# num, bins, patches = plt.hist(p, edgecolor='black', bins=50, density=True, facecolor='g', alpha=0.75)
plt.hist(p, edgecolor='black', bins=50, density=False, facecolor='g', alpha=0.75)
if par == 'mw':
    xaxis = "Molecular weight, kDa"
elif par == "dmax":
    xaxis = "Maximum intra-particle distance, A"
plt.xlabel(xaxis)
plt.ylabel('Number')
aver = round(np.mean(p), 1)
std = round(np.std(p), 1)
minim = round(np.min(p), 1)
maxim = round(np.max(p), 1)
median = round(np.median(p), 1)
# interval95 = np.round(stats.t.interval(0.95, len(p) - 1, aver, std), 1)
interval95 = np.round(np.percentile(p, [2.5, 97.5]), 1)
# print to parse - 1 - mw or dmax 2 - median 3 - percentiles
print(par)
print(median)
print(f"[{interval95[0]} - {interval95[1]}]")

if mType == 'p':
    t = "Globular proteins"
elif mType == 'idp':
    t = "Intrinsically disordered proteins"
elif mType == 'na':
    t = "Nucleic acids"
tt1 = f"{t}:  {par}\n$\mu = {{{aver:.1f}}}; \sigma\ = {{{std:.1f}}}$\n"
tt2 = f"Range: [{minim:.1f}-{maxim:.1f}]\n"
tt3 = f"3$\sigma$ (confidence interval 99.7%):"
tt4 = f"{aver} $\pm\ {{{round(3 * std, 1)}}}$"
tt = tt1 + tt2  # + tt3 + tt4
plt.title(tt)
plt.grid(True)
folder = os.path.dirname(inputFilename)
plt.savefig(f"{folder}/{mType}-{par}-{n}.svg", bbox_inches='tight')