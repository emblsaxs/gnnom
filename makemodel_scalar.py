"""
Train a NN for predicting a scalar value (rg, mw, dmax, v)
and save the model on disc (weights in *.h5 and configuration in *.json files)
"""
import argparse

parser = argparse.ArgumentParser(description='Make NN model - arguments and options.')
parser.add_argument('dataPath', metavar='data', type=str, help='path to the root data folder')
parser.add_argument('logPath', metavar='logs', type=str, help='path to the log folder')
# parser.add_argument('valPath',     metavar='val', default = "", type=str, help='path to the validation data folder')
parser.add_argument('epochs', metavar='epochs', type=int, help='number of epochs')
parser.add_argument('parameter', metavar='parameter', type=str, help='mw/dmax/rg')
parser.add_argument('--units', type=int, default=40, help='number of units in the hidden layer (default: 40)')
parser.add_argument('--first', type=int, default=1, help='index of the first point to use (default: 1)')
parser.add_argument('--last', type=int, default=None, help='index of the last point to use (default: use all)')
parser.add_argument('--weightsPath', '-w', default=None, type=str, help='path to the h5 file')

args = parser.parse_args()

import numpy as np
import saxsdocument
import os
import json
from keras.callbacks import ModelCheckpoint  # , TensorBoard
from keras import optimizers  # , losses
from keras.models import Sequential
from keras.layers import Dense, Activation
from normalisation.logarithm import normalise  # , unnormalise
from utils.crysollog import parseCrysolLogs, readDatsAndLogs

import matplotlib

# AGG backend is for writing to file, not for rendering in a window
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# todo: tf.random.set_seed(5)
import time

start = time.time()

num_epochs = args.epochs
par = args.parameter

# valPath       = args.valPath
dataPath = args.dataPath
logPath = args.logPath
dataFiles = []
valFiles = []

# folders = ["dat-c025"]  # , "dat-c05", "dat-c1", "dat-c2", "dat-c4", "dat-c8", "dat-c16"]
folders = ["dat-c025", "dat-c05", "dat-c1", "dat-c2", "dat-c4", "dat-c8", "dat-c16"]

for f in folders:
    d = os.path.join(dataPath, "training", f)
    v = os.path.join(dataPath, "validation", f)
    fileNames = os.listdir(d)
    valNames = os.listdir(v)
    for ff in fileNames: dataFiles.append(os.path.join(d, ff))
    for ff in valNames: valFiles.append(os.path.join(v, ff))

print("Reading data files...")
# process --first and --last
firstPointIndex = int(args.first) - 1
cur, __ = saxsdocument.read(dataFiles[0])
dat = cur['s']
if args.last:
    if (int(args.last) > len(dat)):
        print(f"--last must be less or equal to the number of points in data files: {args.last}")
        exit()
    lastPointIndex = int(args.last)
else:
    lastPointIndex = len(dat) - 1

smin = dat[firstPointIndex]
smax = dat[lastPointIndex]

Is, logFiles = readDatsAndLogs(dataFiles, logPath, firstPointIndex, lastPointIndex)
IsVal, logFilesVal = readDatsAndLogs(valFiles, logPath, firstPointIndex, lastPointIndex)

print(f"Number of data files found: {len(dataFiles)}")
print(f"Number of log  files found: {len(logFiles)}")
print(f"Number of validation files found: {len(valFiles)}")
print(f"Number of validation log  files found: {len(logFilesVal)}")
print("...done.")

print("Parsing data log files...")
parameters, outCsv = parseCrysolLogs(logFiles, par)
print("...done.")

print("Parsing validation log files...")
parametersVal, outCsvVal = parseCrysolLogs(logFilesVal, par)
outCsv.extend(outCsvVal)
print("...done.")

# save ground true values to csv
outCsvPath = f"ground-{par}-{len(logFiles)}.csv"
np.savetxt(outCsvPath, outCsv, delimiter=",", fmt='%s')
print(outCsvPath + " is written.")

# Perceptron neural network
# tensorboard = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# Number of points in a SAXS curve
N = np.shape(Is)[1]

# Rg, Dmax, MW
output = np.shape(parameters)[1]

# Normalise SAXS input
Is, meanIs, stdIs = normalise(Is)
IsVal, __, __ = normalise(IsVal, meanIs, stdIs)

# #DEBUG
# for I in IsVal:
#     plt.plot(I)
# plt.savefig('validation-norm.png')
# plt.clf()

model = Sequential()
# first layer
# he = np.sqrt(0.06/N)
# model.add(Dense(args.units, input_dim=N, weights = [np.random.uniform(0,he,[args.units, N])]))
model.add(Dense(args.units, input_dim=N, use_bias=True, kernel_initializer='he_uniform'))
# model.add(Dense(input_length, weights = [np.random.uniform(-he,he,[args.bottleneck_units, input_length]), averageIs]))
# model.add(Dense(input_length, weights = [np.zeros([args.hidden_units, input_length]), averageIs]))
model.add(Activation('relu'))

# second layer
model.add(Dense(args.units, use_bias=True, kernel_initializer='he_uniform', bias_initializer='zeros'))
model.add(Activation('relu'))
# third layer
model.add(Dense(args.units, use_bias=True, kernel_initializer='he_uniform', bias_initializer='zeros'))
model.add(Activation('relu'))

avrg = np.mean(parameters)
print(f"Mean {par}: {avrg}")
# marginal imporovement
w = [np.zeros([args.units, 1]), np.array([avrg])]
model.add(Dense(output, weights=w))
model.add(Activation('relu'))
# model.add(Dense(output))
adama = optimizers.Adam(lr=0.001)

model.compile(optimizer=adama, loss='mse')

model_name = f"gnnom-{par}-{firstPointIndex}-{lastPointIndex}-e{args.epochs}-u{args.units}"

if (args.weightsPath):
    model.load_weights(args.weightsPath)

# Check there are no Nans after normalisation
if np.isnan(Is).any():
    print("Error: Is matrix contain Nans!")
    os.exit()
if np.isnan(IsVal).any():
    print("Error: IsVal matrix contain Nans")
    os.exit()

train_history = model.fit(np.array(Is), np.array(parameters), epochs=num_epochs, batch_size=len(dataFiles),
                          validation_data=(np.array(IsVal), np.array(parametersVal)),
                          callbacks=[ModelCheckpoint(model_name + '.h5', save_best_only=True)])

loss = train_history.history['loss']
val_loss = train_history.history['val_loss']

# Confirm that it works
data = np.arange(N)

# save 2d plot of weights in the first layer
plt.imshow(model.get_weights()[0], cmap='coolwarm')
plt.savefig('weights-' + model_name + '.png')
plt.clf()

np.savetxt(f'loss-{model_name}.int', np.transpose(np.vstack((np.arange(num_epochs), loss, val_loss))), fmt="%.8e")

scores = model.evaluate(np.array(IsVal), np.array(parametersVal), verbose=0)
print(f"Metrics: {model.metrics_names}")
print(f"Scores: {scores}")

# serialize model to JSON
model_str = model.to_json()
model_json = json.loads(model_str)
model_json['smin'] = smin
model_json['smax'] = smax
model_json['firstPointIndex'] = firstPointIndex  # including, starts from 0
model_json['lastPointIndex'] = lastPointIndex  # excluding
model_json['meanIs'] = list(meanIs)
model_json['stdIs'] = list(stdIs)
# model_json['KratkyDegree']    = args.degree
# compute elapsed time
end = time.time()
t = str(round((end - start) / 60, 2))
model_json['minutesTrained'] = t

with open(model_name + ".json", "w") as json_file:
    json_file.write(json.dumps(model_json))
# serialize weights to HDF5
# model.save_weights(model_name + ".h5") #last but not best weights
print("Saved model " + model_name + " to disk")
