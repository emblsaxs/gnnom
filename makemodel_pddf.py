#!/usr/bin/env python
# coding: utf-8


import keras
import numpy as np
import saxsdocument
import os
import sys
import json

from keras.callbacks import TensorBoard
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation

# Do not use Xwindows backend
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# call like this:
# python makemodels_pddf.py 10000 /(epochs)
num_epochs = int(sys.argv[1])

dataPath = "/media/NewBig/ownCloud/embl/DARA/p-of-r-2/NN/jupyter-notebooks/big_data-top-1023"
PDDFPath = "/media/NewBig/ownCloud/embl/DARA/p-of-r-2/NN/jupyter-notebooks/big_pddf-top-1023"

#dataPath = "/afs/embl-hamburg.de/groups/saxs/Al/DARA/p-of-r-2/python/int-with-noise-ler8192"
#PDDFPath = "/afs/embl-hamburg.de/groups/saxs/Al/DARA/p-of-r-2/python/pddf-extrapolated-ler8192"

# Make sure there are no subfolders!
dataFiles = os.listdir(dataPath)
PDDFFiles = os.listdir(PDDFPath)

dataFiles.sort()
PDDFFiles.sort()

print("Number of data files found: " + str(len(dataFiles)))
print("Number of PDDF files found: " + str(len(PDDFFiles)))

# Complain if different number of Is and PDDF files was read
if (len(dataFiles) != len(PDDFFiles)):
    print("Warning: there is a mismatch between data files and PDDF files!!!")

n_all   = len(PDDFFiles)
n_cases = (int)(0.95 * n_all)

print("Reading data files...")

Is = []

for file in dataFiles:
    path = os.path.join(dataPath, file)
    doc  = saxsdocument.read(path)
    dat  = np.transpose(np.array(doc.curve[0]))
    Is.append(dat[1])

print("...done.")


print("Reading PDDF files...")

# Define max number of points in output p(r); fill missing points with zeroes
output_length = 301

PDDF = []

for file in PDDFFiles:
    path = os.path.join(PDDFPath, file)
    doc  = saxsdocument.read(path)
    dat  = np.transpose(np.array(doc.curve[0]))[1]
    negIndex = np.argmax(dat < 0)
    dat[negIndex:] = 0.0
    PDDF.append(dat)

print("...done.")


# NB: without p(r) normalization the result is much worse

# Normalize PDDF: linear, from 0 to 1:
#maxPDDF = np.max(PDDF)
#PDDF = PDDF / maxPDDF

# Normalize PDDF: subtract mean, divide by stdev:
#meanPDDF = np.mean(PDDF)
#stdPDDF  = np.std(PDDF)
#PDDF = PDDF - meanPDDF
#PDDF = PDDF / stdPDDF

# Normalize PDDF: divide by stdev:
stdPDDF  = np.std(PDDF)
PDDF = PDDF / stdPDDF


# Perceptron neural network
tensorboard = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# Number of points in a SAXS curve
N = np.shape(Is)[1]

# Number of points in a PDDF curve
output = output_length

model = Sequential()
# first layer
model.add(Dense(80, input_dim=N, use_bias=True, kernel_initializer='he_uniform', bias_initializer='zeros'))
model.add(Activation('relu'))

# second layer
model.add(Dense(output))

model.compile(optimizer='adam', loss='mse')

train_history = model.fit(np.array(Is[0:n_cases]), np.array(PDDF[0:n_cases]), epochs=num_epochs,  batch_size=128,
                          validation_data =  (np.array(Is[n_cases:n_all]), np.array(PDDF[n_cases:n_all]))
                          , callbacks = [tensorboard])

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

# Heat map of neuron weights
plt.imshow(model.get_weights()[0], vmin=-1, vmax=1, cmap='coolwarm')
plt.savefig('weights-0.png')
plt.clf()

num = 41

# writes 20 PDDF fit plots
for i in range(num, num + 20):
    test = np.array([Is[i], ])
    pred = model.predict(test)
    plot = plt.plot(np.transpose(pred))
    plt.plot(PDDF[i])
    plt.savefig('pddf_' + str(i) +'.png')
    plt.clf()

model_name = "gnnom_pddf"

# evaluate the model
scores = model.evaluate(np.array(Is[0:n_cases]), np.array(PDDF[0:n_cases]), verbose=0)

print(model.metrics_names)
print(scores)
 
# serialize model to JSON
model_str = model.to_json()
model_json = json.loads(model_str)
model_json['Normalization coefficient'] = stdPDDF

with open(model_name + ".json", "w") as json_file:
    json_file.write(json.dumps(model_json))
# serialize weights to HDF5
model.save_weights(model_name + ".h5")
print("Saved model " + model_name + " to disk")
