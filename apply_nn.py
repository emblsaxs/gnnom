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
parser.add_argument('--units', type=str, default='NANOMETER', help='angular units: ANGSTROM or NANOMETER')
parser.add_argument('--n', default=1000, type=int, help='how many times to resample')
# parser.add_argument('-o', '--output', type=str, default="", help='prefix to output CSV files')

args = parser.parse_args()

from keras.models import model_from_json
from mysaxsdocument import saxsdocument
import numpy as np
import json
import os
import sys
from normalisation.meanvariance import normalise
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

set_matplotlib_formats('svg')

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
if units not in ['ANGSTROM', 'NANOMETER']:
    parser.error("Wrong units! Please choose between ANGSTROM and NANOMETER.")
n = args.n
inputFilename = args.dataPath
I0 = args.I0
# read saxs data, find smin and smax
try:
    cur, __ = saxsdocument.read(inputFilename)
    s = cur['s']
    if units == "NANOMETER": s = [ss / 10.0 for ss in s]  # change to angstroms
    smin = s[0]
    smax = s[-1]
    if smin > smin0:
        print(f"Error: Insufficient angular range!"
              f"smin = {smin} > {smin0} A^-1")
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
    Err = cur['Err']

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
    if 'Normalization coefficient' in json_data:
        multiplier = float(json_data['Normalization coefficient'])
    if 'meanIs' in json_data:
        meanIs = json_data['meanIs']
        stdIs = json_data['stdIs']
    elif 'meanIs' not in json_data:
        print(f"WARNING! "
              f"{jsonFilename} does not contain normalization coefficients!"
              f"Proceeding without normalization...")
    # Compulsory fields in json
    smin = (float)(json_data['smin'])
    smax = (float)(json_data['smax'])
    firstPointIndex = (int)(json_data['firstPointIndex'])
    lastPointIndex = (int)(json_data['lastPointIndex'])

    jsonFile.close()
    loadedModel = model_from_json(loadedModelJson)
    # load weights into new model
    h5Filename = modelPath + ".h5"
    loadedModel.load_weights(h5Filename)
    inputLength = loadedModel.input_shape[1]  # I(s) points
    print(f"Expected input: {inputLength} points.")
    # outputLength = loadedModel.output_shape[1]  # p(r) points
    print("Model loaded. Yeah!")

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

# resample n times and run nn to do prediction
p = []
for i in range(n):
    Is = np.random.normal(INew, ErrNew)
    try:
        Is, __, __ = normalise(Is, stdIs, meanIs)
    except:
        pass

    test = np.array([Is, ])
    pred = loadedModel.predict(test)

    for number in pred[0]:
        p.append(round(multiplier * number, 3))
print(f"{par} mean : {round(np.mean(p), 2)}\n"
      f"{par} std  : {round(np.std(p), 2)}")

# num, bins, patches = plt.hist(p, edgecolor='black', bins=50, density=True, facecolor='g', alpha=0.75)
plt.hist(p, edgecolor='black', bins=50, density=False, facecolor='g', alpha=0.75)
plt.xlabel(f"{par}")
plt.ylabel('Number')
aver = round(np.mean(p), 2)
minim = round(np.min(p), 2)
maxim = round(np.max(p), 2)
if mType == 'p':
    t = "Globular proteins"
elif mType == 'idp':
    t = "Intrinsically disordered proteins"
elif mType == 'na':
    t = "Nucleic acids"
tt = f"{t}:  {par}\nMean: {aver}\nMin: {minim}  Max: {maxim}"
plt.title(tt)
plt.grid(True)
plt.savefig(f"{mType}-{par}-{n}.svg", bbox_inches='tight')
