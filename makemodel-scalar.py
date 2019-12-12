#!/usr/bin/python
# coding: utf-8

import argparse

parser = argparse.ArgumentParser(description='Make NN model - arguments and options.')
parser.add_argument('dataPath', metavar='data',   type=str, help='path to the training data folder')
parser.add_argument('logPath',  metavar='logs',   type=str, help='path to the training log folder')
parser.add_argument('epochs',   metavar='epochs', type=int, help='number of epochs')
parser.add_argument('--units', type=int, default=40, help='number of units in the hidden layer (default: 40)')
parser.add_argument('--first', type=int, default=1,  help='index of the first point to use (default: 1)')
parser.add_argument('--last',  type=int, default=-1, help='index of the last point to use (default: use all)')

args = parser.parse_args()


import keras
import numpy as np
import saxsdocument
import os
from keras.callbacks import TensorBoard, ModelCheckpoint

from keras.models import Sequential
from keras.layers import Dense, Activation

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

#np.random.seed(5)
#tf.random.set_seed(5)


num_epochs = int(args.epochs)

# Make sure there are no subfolders!
dataFiles = os.listdir(args.dataPath)
logFiles = os.listdir(args.logPath)

dataFiles.sort()
logFiles.sort()

print("Number of data files found: " + str(len(dataFiles)))
print("Number of log  files found: " + str(len(logFiles)))

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
    print("--last must be less or equal to the number of points in data files: " + str(lastPointIndex))
    exit()

if(args.last != -1):
    lastPointIndex = int(args.last)

    

for file in dataFiles:
    path = os.path.join(args.dataPath, file)
    doc  = saxsdocument.read(path)
    dat  = np.transpose(np.array(doc.curve[0]))
    Is.append(dat[1][firstPointIndex:lastPointIndex])

print("...done.")


print("Reading log files...")
parameters = []

for file in logFiles:
    path = os.path.join(args.logPath, file)
    lines = [line.strip() for line in open(path)]
    rgdmaxmw = []
    # Read 'Molecular Weight: 0.4330E+06':
    for line in lines:
        if "slope" in line:
            rgdmaxmw.append(line.split()[-1])
            parameters.append(rgdmaxmw)
            break


#        if "Weight" in line:
#            rgdmaxmw.append(line.split()[2])
#            parameters.append(rgdmaxmw)
#            break

print("...done.")



# Perceptron neural network
tensorboard = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

####
# Number of points in a SAXS curve
N = np.shape(Is)[1]

# Rg, Dmax, MW
output = np.shape(parameters)[1]


model = Sequential()
# first layer
model.add(Dense(args.units, input_dim=N, use_bias=True, kernel_initializer='he_uniform', bias_initializer='zeros'))
model.add(Activation('relu'))

# second layer
model.add(Dense(args.units, use_bias=True, kernel_initializer='he_uniform', bias_initializer='zeros'))
model.add(Activation('relu'))

model.add(Dense(output))

model.compile(optimizer='adam', loss='mse')

train_history = model.fit(np.array(Is[0:n_cases]), np.array(parameters[0:n_cases]), epochs=num_epochs,  batch_size=32,
                          validation_data =  (np.array(Is[n_cases:n_all]), np.array(parameters[n_cases:n_all])),
                          callbacks = [tensorboard, ModelCheckpoint('best.h5', save_best_only=True)])




model_name = "gnnom-avrg-i-rg-001-04-e" + str(args.epochs) + "-u" + str(args.units)


loss = train_history.history['loss']
val_loss = train_history.history['val_loss']
plt.semilogy(loss)
plt.semilogy(val_loss)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss: ' +str(len(Is[0:n_cases])) + ' frames',
            'validation loss: ' + str(len(Is[n_cases:n_all])) + ' frames'])
plt.title('final loss: ' + str(train_history.history['loss'][-1]) +
          '\nfinal val_loss: ' + str(train_history.history['val_loss'][-1]))

plt.savefig('loss-' + model_name + '.png', format='png', dpi=250)
plt.clf()
# Confirm that it works
data = np.arange(N)

plt.imshow(model.get_weights()[0], cmap='coolwarm')
plt.savefig('weights-' + model_name + '.png')
plt.clf()

scores = model.evaluate(np.array(Is[0:n_cases]), np.array(parameters[0:n_cases]), verbose=0)

print(model.metrics_names)
print(scores)
 
# serialize model to JSON
model_json = model.to_json()
with open(model_name + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(model_name + ".h5")
print("Saved model " + model_name + " to disk")

#print("average Rg over the learning set:   " + str(avrgRg))
