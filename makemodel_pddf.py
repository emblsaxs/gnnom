#!/usr/bin/python
# coding: utf-8

import argparse

parser = argparse.ArgumentParser(description='Make NN model - arguments and options.')
parser.add_argument('dataPath', metavar='data', type=str, help='path to the training data folder')
parser.add_argument('pddfPath', metavar='pddf', type=str, help='path to the training pddf folder')
parser.add_argument('--epochs', default=None,   type=int, help='number of epochs')
parser.add_argument('--layers', default=2,      type=int, help='number of hidden layers')
parser.add_argument('--units',  default=80,     type=int, help='number of units in the hidden layer (default: 40)')
parser.add_argument('--first',   type=int, default= 1, help='index of the first point to use (default: 1)')
parser.add_argument('--last',    type=int, default=None, help='index of the last point to use (default: use all)')
parser.add_argument('--valData', type=str, help='path to the validation data folder')
parser.add_argument('--valPddf', type=str, help='path to the validation pddf folder')
parser.add_argument('-p', '--prefix', type=str, default='', help='prefix for the output file names')
parser.add_argument('--weightsPath', '-w', default=None, type=str, help='path to the h5 file')

args = parser.parse_args()

import keras
import numpy as np
import psaxsdocument as saxsdocument
import time
import os
import json

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import losses, optimizers

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import time

start = time.time()

def readFiles(path, isPddf = False, firstPointIndex = 0, lastPointIndex = None):
    # Make sure there are no subfolders!
    files = os.listdir(path)
    files.sort()
    m = "data"
    if (isPddf): m = "PDDF"
    arr = []
    for f in files:
        p = os.path.join(path, f)
        #doc  = saxsdocument.read(p)
        #dat  = np.transpose(doc.curve[0])[1]
        __, cur = saxsdocument.read(p)
        dat = np.array(cur['I'])
        if isPddf:
            # Nullify all points after the first p(r) < 0
            negIndex = np.argmax(dat < 0)
            dat[negIndex:] = 0.0
            arr.append(dat)
        else:
            arr.append(dat[firstPointIndex:lastPointIndex])
    print(f"{path}: {len(arr)} {m} files have been read")
    return np.array(arr)


# Process --first and --last:
firstPointIndex = args.first - 1
lastPointIndex  = args.last

folders = ["dat-c025", "dat-c05", "dat-c1", "dat-c2", "dat-c4", "dat-c8", "dat-c16"]
#folders = ["dat-c16", "dat-c8"]

Is   = []
pddf = []
# read training data and pddf
for f in folders:
    d = os.path.join(args.dataPath,f)
    I = readFiles(d, False, firstPointIndex, lastPointIndex)
    Is.extend(I)
    p = readFiles(args.pddfPath, True)
    pddf.extend(p)

Is   = np.array(Is)
pddf = np.array(pddf)
n_all = len(Is)

# determine smin and smax from the first saxs file
path = os.path.join(args.dataPath,folders[0])
filesData = os.listdir(path)
firstFile = os.path.join(path, filesData[0])
__, cur = saxsdocument.read(firstFile)
s = cur['s']

smin = s[firstPointIndex]
if (args.last): smax = s[args.last - 1]
else: smax = s[-1]
print(f"smin {smin} smax {smax}")

# determine rmin and rmax from the first pddf file
pddfData = os.listdir(args.pddfPath)
firstFile = os.path.join(args.pddfPath, pddfData[0])
__, cur = saxsdocument.read(firstFile)
r = cur['s']

rmin = r[0]
rmax = r[-1]
pddfNumberOfPoints = len(r)
print(f"rmin {rmin} rmax {rmax}")

# read validation data and pddf
if args.valData:
    for f in folders:
        d = os.path.join(args.valData,f)
        #read training set files
        IsVal        = readFiles(d, False, firstPointIndex, lastPointIndex)
        pddfVal      = readFiles(args.valPddf, True)
        pddfVal      = pddfVal / np.std(pddfVal)
        n_val        = len(IsVal)
else:
    n_cases = int(n_all * 0.9)
    IsVal   =   Is[n_cases:n_all]
    pddfVal = pddf[n_cases:n_all]
    Is      =   Is[0:n_cases]
    pddf    = pddf[0:n_cases]

stdpddf = np.std(pddf)
pddf    = pddf / stdpddf
pddfVal = pddfVal/np.std(pddfVal)
pddfMean  = np.mean(pddf, axis = 0)

print(f"Total: {len(Is)} training data files")
print(f"Total: {len(pddf)} training PDDF files")
print(f"Total: {len(IsVal)} validation data files")
print(f"Total: {len(pddfVal)} validation PDDF files")
    

if args.epochs:
    num_epochs = int(args.epochs)
else:
    num_epochs = int(20000000.0 / len(Is))

# Number of points in a SAXS curve
input_length  = np.shape(Is)[1]
# Number of points in pddf
output_length = np.shape(pddf)[1]

model = Sequential()

# first layer
model.add(Dense(args.units, input_dim=input_length, use_bias=True, kernel_initializer='he_uniform', bias_initializer='zeros'))
model.add(Activation('tanh'))

for layer in range(args.layers - 1):
   # add layer
   model.add(Dense(args.units, use_bias=True, kernel_initializer='he_uniform', bias_initializer='zeros'))
   model.add(Activation('tanh'))

# output layer
w = [np.zeros([args.units, len(pddfMean)]), pddfMean]
model.add(Dense(output_length, weights = w))

adama = optimizers.Adam(lr=0.0001)
#adama = optimizers.Adam(lr=0.00001)

model.compile(optimizer= adama, loss='mse')

model_name = f"gnnom-pddf-{args.prefix}-e{num_epochs}-u{args.units}-l{args.layers}"

if(args.weightsPath):
    model.load_weights(args.weightsPath)

train_history = model.fit(Is, pddf, epochs=num_epochs,  batch_size=n_all,
                          validation_data =  (IsVal, pddfVal),
                          callbacks = [ModelCheckpoint(model_name + '.h5', save_best_only=True)])


loss = train_history.history['loss']
val_loss = train_history.history['val_loss']

#plt.semilogy(loss)
#plt.semilogy(val_loss)
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['training loss: ' +str(len(Is[0:n_cases])) + ' frames',
#            'validation loss: ' + str(len(Is[n_cases:n_all])) + ' frames'])
#plt.title('final loss: ' + str(train_history.history['loss'][-1]) +
#          '\nfinal val_loss: ' + str(train_history.history['val_loss'][-1]))
#
#plt.savefig('loss-' + model_name + '.png', format='png', dpi=250)
#plt.clf()


data = np.arange(input_length)

#plt.imshow(model.get_weights()[0], cmap='coolwarm')
#plt.savefig('weights-' + model_name + '.png')
#plt.clf()

np.savetxt('loss-' + model_name + '.int', np.transpose(np.vstack((np.arange(num_epochs),loss, val_loss))), fmt = "%.8e")

# compute consumed time
end = time.time()
t   = str(round((end - start) / 60,2))
# serialize model to JSON
model_str = model.to_json()
model_json = json.loads(model_str)
model_json['Normalization coefficient'] = stdpddf
model_json['smin'] = smin
model_json['smax'] = smax
model_json['firstPointIndex'] = firstPointIndex  # including, starts from 0
if not lastPointIndex: lastPointIndex = len(s)
model_json['lastPointIndex']  = lastPointIndex   # excluding
model_json['rmin'] = rmin
model_json['rmax'] = rmax
model_json['pddfNumberOfPoints'] = pddfNumberOfPoints
model_json['minutesTrained']    = t   

with open(model_name + ".json", "w") as json_file:
    json_file.write(json.dumps(model_json))
# serialize weights to HDF5
#model.save_weights(model_name + ".h5") #last but not best weights
print(f"Saved model {model_name} to disk")
print(f"stdpddf: {stdpddf}")

