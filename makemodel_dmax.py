#!/usr/bin/env python
# coding: utf-8


import keras
import numpy as np
import saxsdocument
import os
import sys
from keras.callbacks import TensorBoard
import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, Conv1D, MaxPooling1D, ZeroPadding1D, Dropout

# Do not use Xwindows backend
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

num_epochs = int(sys.argv[1])

# call like this:
# python makemodels_rg.py 10000 /(epochs)

dataPath = "learning_sets/big_data-top-1023"
logPath  = "learning_sets/big_log-top-1023"

# Make sure there are no subfolders!
dataFiles = os.listdir(dataPath)
logFiles = os.listdir(logPath)

dataFiles.sort()
logFiles.sort()

print("Number of data files found: " + str(len(dataFiles)))
print("Number of log  files found: " + str(len(logFiles)))

n_all   = len(dataFiles)
n_cases = 923

print("Reading data files...")

Is = []

for file in dataFiles:
    path = os.path.join(dataPath, file)
    doc  = saxsdocument.read(path)
    dat  = np.transpose(np.array(doc.curve[0]))
    Is.append(dat[1])

print("...done.")


print("Reading log files...")
# We assume Rg, Dmax and MW were grepped from *.log.
# Contains Rg[A], Dmax[A], MW[Da]:
parameters = []

for file in logFiles:
    path = os.path.join(logPath, file)
    lines = [line.strip() for line in open(path)]
    rgdmaxmw = []
    # Read 'Rg from the slope of net intensity':
    #rgdmaxmw.append(float(lines[0].split()[-1]))# / 21.2)
    # Read 'Envelope  diameter':
    rgdmaxmw.append(float(lines[1].split()[-1]))# / 69.1)
    # Read 'Molecular Weight':
    #rgdmaxmw.append(float(lines[2].split()[2]) / 1000.0 / 35.0)
    parameters.append(rgdmaxmw)

print("...done.")

# Perceptron neural network
tensorboard = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# Number of points in a SAXS curve
N = np.shape(Is)[1]

# Rg, Dmax, MW
output = np.shape(parameters)[1]


model = Sequential()
# first layer
model.add(Dense(80, input_dim=N, use_bias=True, kernel_initializer='he_uniform', bias_initializer='zeros'))
model.add(Activation('relu'))

# second layer
model.add(Dense(output))

model.compile(optimizer='adam', loss='mse')

train_history = model.fit(np.array(Is[0:n_cases]), np.array(parameters[0:n_cases]), epochs=num_epochs,  batch_size=32,
                          validation_data =  (np.array(Is[n_cases:n_all]), np.array(parameters[n_cases:n_all])),
                          callbacks = [tensorboard])

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

plt.savefig('loss_e'+str(num_epochs)+'.png',format='png', dpi=250)
plt.clf()
# Confirm that it works
data = np.arange(N)

plt.imshow(model.get_weights()[0], vmin=-1, vmax=1, cmap='coolwarm')
plt.savefig('weights-0.png')
plt.clf()

model_name = "gnnom_dmax"

# evaluate the model
scores = model.evaluate(np.array(Is[0:n_cases]), np.array(parameters[0:n_cases]), verbose=0)
print(model.metrics_names)
print(scores)
 
# serialize model to JSON
model_json = model.to_json()
with open(model_name + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(model_name + ".h5")

print("Saved model " + model_name +" to disk")

