#!/usr/bin/python
# coding: utf-8

import argparse

parser = argparse.ArgumentParser(description='Make NN model - arguments and options.')
parser.add_argument('dataPath', metavar='data', type=str, help='path to the training data folder')
parser.add_argument('pddfPath', metavar='pddf', type=str, help='path to the training pddf folder')
parser.add_argument('--epochs', default=None,   type=int, help='number of epochs')
parser.add_argument('--layers', default=2,      type=int, help='number of hidden layers')
parser.add_argument('--units',  default=40,     type=int, help='number of units in the hidden layer (default: 40)')
parser.add_argument('--first',   type=int, default= 1, help='index of the first point to use (default: 1)')
parser.add_argument('--last',    type=int, default=-1, help='index of the last point to use (default: use all)')
parser.add_argument('--valData', type=str, help='path to the validation data folder')
parser.add_argument('--valPddf', type=str, help='path to the validation pddf folder')
parser.add_argument('-p', '--prefix', type=str, default='', help='prefix for the output file names')

args = parser.parse_args()


import keras
import numpy as np
import saxsdocument
#from saxsdocument import pysaxsdocument as saxsdocument
import os
import json

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def readFiles(path, isPddf = False, firstPointIndex = 0, lastPointIndex = None):
    # Make sure there are no subfolders!
    files = os.listdir(path)
    files.sort()
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
    return np.array(arr)


# Process --first and --last:
firstPointIndex = int(args.first) - 1

# Read first I(s) file to get number of points
file = os.listdir(args.dataPath)[0]
path = os.path.join(args.dataPath, file)
__, doc  = saxsdocument.read(path)
#dat  = doc.curve[0]
s   = doc['s']

lastPointIndex = len(s)
if(int(args.last) > lastPointIndex):
    print(f"--last must be less or equal to the number of points in data files: {lastPointIndex}")
    exit()
if(args.last != -1):
    lastPointIndex = int(args.last)

smin = s[firstPointIndex]
smax = s[lastPointIndex - 1]

#read first pddf file to get number of points
file          = os.listdir(args.pddfPath)[0]
path          = os.path.join(args.pddfPath, file)
__, doc       = saxsdocument.read(path)
dat           = doc['I']#.curve[0]
output_length = len(dat)

#read training set files
Is        = readFiles(args.dataPath, False, firstPointIndex, lastPointIndex)
pddf      = readFiles(args.pddfPath, True)
pddfMean  = np.mean(pddf, axis = 0)
n_all     = len(Is)
# NB: without p(r) normalization the result is much worse
# Normalize PDDF: divide by stdev:
stdpddf  = np.std(pddf)
pddf = pddf / stdpddf

#create validation set
if args.valData and args.valPddf:
    IsVal   = readFiles(args.valData, False, firstPointIndex, lastPointIndex)
    pddfVal = readFiles(args.valPddf, True)
    # NB: without p(r) normalization the result is much worse
    # Normalize PDDF: divide by stdev:
    pddfVal = pddfVal / stdpddf

else:
    n_cases = int(n_all * 0.9)
    IsVal   =   Is[n_cases:n_all]
    pddfVal = pddf[n_cases:n_all]
    Is      =   Is[0:n_cases]
    pddf    = pddf[0:n_cases]

if args.epochs:
    num_epochs = int(args.epochs)
else:
    num_epochs = int(20000000.0 / len(Is))



#tensorboard = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# Number of points in a SAXS curve
input_length  = np.shape(Is)[1]

model = Sequential()

# first layer
model.add(Dense(args.units, input_dim=input_length, use_bias=True, kernel_initializer='he_uniform', bias_initializer='zeros'))
model.add(Activation('relu'))

for layer in range(args.layers - 1):
   # add layer
   model.add(Dense(args.units, use_bias=True, kernel_initializer='he_uniform', bias_initializer='zeros'))
   model.add(Activation('relu'))

# output layer
w = [np.zeros([args.units, len(pddfMean)]), pddfMean]
model.add(Dense(output_length, weights = w))

model.compile(optimizer='adam', loss='mse')

model_name = f"gnnom-i0-pddf-{args.prefix}-e{num_epochs}-u{args.units}-l{args.layers}"

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

plt.imshow(model.get_weights()[0], cmap='coolwarm')
plt.savefig('weights-' + model_name + '.png')
plt.clf()

np.savetxt('loss-' + model_name + '.int', np.transpose(np.vstack((np.arange(num_epochs),loss, val_loss))), fmt = "%.8e")

 
# serialize model to JSON
model_str = model.to_json()
model_json = json.loads(model_str)
model_json['Normalization coefficient'] = stdpddf
model_json['smin'] = smin
model_json['smax'] = smax
model_json['firstPointIndex'] = firstPointIndex  # including, starts from 0
model_json['lastPointIndex']  = lastPointIndex   # excluding

with open(model_name + ".json", "w") as json_file:
    json_file.write(json.dumps(model_json))
# serialize weights to HDF5
#model.save_weights(model_name + ".h5") #last but not best weights
print(f"Saved model {model_name} to disk")
print(f"stdpddf: {stdpddf}")

