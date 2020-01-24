#!/usr/bin/python
# coding: utf-8

import argparse

parser = argparse.ArgumentParser(description='Make NN model - arguments and options.')
parser.add_argument('dataPath', metavar='data',   type=str, help='path to the training data folder')
parser.add_argument('pddfPath', metavar='pddf',   type=str, help='path to the training pddf folder')
parser.add_argument('epochs',   metavar='epochs', type=int, help='number of epochs')
parser.add_argument('--units', type=int, default=40, help='number of units in the hidden layer (default: 40)')
parser.add_argument('--first', type=int, default=1,  help='index of the first point to use (default: 1)')
parser.add_argument('--last',  type=int, default=-1, help='index of the last point to use (default: use all)')

args = parser.parse_args()


import keras
import numpy as np
import saxsdocument
import os
import json

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

num_epochs = int(args.epochs)

# Make sure there are no subfolders!
dataFiles = os.listdir(args.dataPath)
pddfFiles = os.listdir(args.pddfPath)

dataFiles.sort()
pddfFiles.sort()

print(f"Number of data files found: {len(dataFiles)}")
print(f"Number of pddf files found: {len(pddfFiles)}")

n_all   = len(dataFiles)
n_cases = int(n_all * 0.9)

print("Reading data files...")

Is = []


# Process --first and --last:
firstPointIndex = int(args.first) - 1

path = os.path.join(args.dataPath, dataFiles[0])
doc  = saxsdocument.read(path)
dat  = np.array(doc.curve[0])

lastPointIndex = len(dat)
if(int(args.last) > lastPointIndex):
    print(f"--last must be less or equal to the number of points in data files: {lastPointIndex}")
    exit()
if(args.last != -1):
    lastPointIndex = int(args.last)

smin = dat[firstPointIndex][0]
smax = dat[lastPointIndex - 1][0]

for file in dataFiles:
    path = os.path.join(args.dataPath, file)
    doc  = saxsdocument.read(path)
    dat  = np.transpose(np.array(doc.curve[0]))
    Is.append(dat[1][firstPointIndex:lastPointIndex])

print("...done.")


print("Reading pddf files...")
pddf = []

for file in pddfFiles:
    path = os.path.join(args.pddfPath, file)
    doc  = saxsdocument.read(path)
    dat  = np.transpose(np.array(doc.curve[0]))[1]
    # Nullify all points after the first p(r) < 0
    negIndex = np.argmax(dat < 0)
    dat[negIndex:] = 0.0
    pddf.append(dat)

print("...done.")

# NB: without p(r) normalization the result is much worse
# Normalize PDDF: divide by stdev:
stdpddf  = np.std(pddf)
pddf = pddf / stdpddf


#tensorboard = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# Number of points in a SAXS curve
input_length  = np.shape(Is)[1]

#FIXME: read output_length from training pddf data
# Number of points in a p(r)
output_length = 401

model = Sequential()

# first layer
model.add(Dense(args.units, input_dim=input_length, use_bias=True, kernel_initializer='he_uniform', bias_initializer='zeros'))
model.add(Activation('relu'))

# second layer
model.add(Dense(args.units, use_bias=True, kernel_initializer='he_uniform', bias_initializer='zeros'))
model.add(Activation('relu'))

# output layer
model.add(Dense(output_length))

model.compile(optimizer='adam', loss='mse')

model_name = "gnnom-i0-pddf-e" + str(args.epochs) + "-u" + str(args.units) + "-l2"

train_history = model.fit(np.array(Is[0:n_cases]), np.array(pddf[0:n_cases]), epochs=num_epochs,  batch_size=32,
                          validation_data =  (np.array(Is[n_cases:n_all]), np.array(pddf[n_cases:n_all])),
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

