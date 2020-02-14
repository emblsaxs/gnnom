#!/usr/bin/python
# coding: utf-8

import argparse

parser = argparse.ArgumentParser(description='Make NN autoencoder model - arguments and options.')
parser.add_argument('dataPath', metavar='data', type=str, help='path to the training data folder')
parser.add_argument('--weightsPath', '-w', default=None, type=str, help='path to the h5 file')
parser.add_argument('--epochs', default=None,   type=int, help='number of epochs')
parser.add_argument('--bottleneck_units', '-bu', default=10,  type=int, help='number of units in bottleneck layer (default: 10)')
parser.add_argument('--hidden_units', '-hu',  default=80,     type=int, help='number of units in the hidden layer (default: 80)')
parser.add_argument('--first',   type=int, default= 1, help='index of the first point to use (default: 1)')
parser.add_argument('--last',    type=int, default=-1, help='index of the last point to use (default: use all)')
parser.add_argument('--degree',  type=float, default= 0.0, help='I = I*s^degree (default: 0)')
parser.add_argument('--valData', type=str, help='path to the validation data folder')
parser.add_argument('-p', '--prefix', type=str, default='', help='prefix for the output file names')

args = parser.parse_args()


import numpy as np
import saxsdocument
import os
import json

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import losses

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def readFiles(path, firstPointIndex = 0, lastPointIndex = None, degree = 0):
    # Make sure there are no subfolders!
    files = os.listdir(path)
    files.sort()
    arr = []
    for f in files:
        p = os.path.join(path, f)
        doc  = saxsdocument.read(p)
        s    = np.transpose(doc.curve[0])[0]
        dat  = np.transpose(doc.curve[0])[1]
        arr.append(dat[firstPointIndex:lastPointIndex]*s**(degree))
    return np.array(arr)


# Process --first and --last:
firstPointIndex = int(args.first) - 1

# Read first I(s) file to get number of points
file = os.listdir(args.dataPath)[0]
path = os.path.join(args.dataPath, file)
doc  = saxsdocument.read(path)
dat  = doc.curve[0]

lastPointIndex = len(dat)
if(int(args.last) > lastPointIndex):
    print(f"--last must be less or equal to the number of points in data files: {lastPointIndex}")
    exit()
if(args.last != -1):
    lastPointIndex = int(args.last)

smin = dat[firstPointIndex][0]
smax = dat[lastPointIndex - 1][0]

#read training set files
Is   = readFiles(args.dataPath, firstPointIndex, lastPointIndex, args.degree)
n_all = len(Is)

#create validation set
if args.valData:
    IsVal   = readFiles(args.valData, firstPointIndex, lastPointIndex, args.degree)

else:
    n_cases = int(n_all * 0.9)
    IsVal   =   Is[n_cases:n_all]
    Is      =   Is[0:n_cases]

if args.epochs:
    num_epochs = int(args.epochs)
else:
    num_epochs = int(20000000.0 / len(Is))


#tensorboard = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# Number of points in a SAXS curve
input_length  = np.shape(Is)[1]

model = Sequential()

# first layer
model.add(Dense(args.bottleneck_units, input_dim=input_length, use_bias=True, kernel_initializer='he_uniform', bias_initializer='zeros'))
model.add(Activation('relu'))


# first layer
#model.add(Dense(args.hidden_units, input_dim=input_length, use_bias=True, kernel_initializer='he_uniform', bias_initializer='zeros'))
#model.add(Activation('relu'))

#for layer in range(args.layers - 1):
#   # add layer
#   model.add(Dense(args.units, use_bias=True, kernel_initializer='he_uniform', bias_initializer='zeros'))
#   model.add(Activation('relu'))


#model.add(Dense(args.bottleneck_units, use_bias=True, kernel_initializer='he_uniform', bias_initializer='zeros'))
#model.add(Activation('relu'))

#model.add(Dense(args.hidden_units, use_bias=True, kernel_initializer='he_uniform', bias_initializer='zeros'))
#model.add(Activation('relu'))

# output layer
model.add(Dense(input_length))

if(args.weightsPath):
    model.load_weights(args.weightsPath)

model.compile(optimizer='adam', loss=losses.huber_loss)

model_name = f"autoencoder-{args.prefix}-e{num_epochs}-bu{args.bottleneck_units}-l1-d{args.degree}"

train_history = model.fit(Is, Is, epochs=num_epochs,  batch_size=32,
                          validation_data =  (IsVal, IsVal),
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
model_json['smin'] = smin
model_json['smax'] = smax
model_json['firstPointIndex'] = firstPointIndex  # including, starts from 0
model_json['lastPointIndex']  = lastPointIndex   # excluding
model_json['KratkyDegree']    = args.degree

with open(model_name + ".json", "w") as json_file:
    json_file.write(json.dumps(model_json))
# serialize weights to HDF5
#model.save_weights(model_name + ".h5") #last but not best weights
print(f"Saved model {model_name} to disk")

