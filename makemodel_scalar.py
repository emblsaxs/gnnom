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
parser.add_argument('--units', type=int, default=80, help='number of units in the hidden layer (default: 40)')
parser.add_argument('--first', type=int, default=1, help='index of the first point to use (default: 1)')
parser.add_argument('--last',  type=int, default=None, help='index of the last point to use (default: use all)')
parser.add_argument('--weightsPath', '-w', default=None, type=str, help='path to the h5 file')
parser.add_argument('--picklePath', '-p', default=None, type=str, help='path to the pickle file, by default data.p')

args = parser.parse_args()

import numpy as np
import saxsdocument
import os
import json
from keras.callbacks import ModelCheckpoint  # , TensorBoard
from keras import optimizers  # , losses
from keras.models import Sequential
from keras.layers import Dense, Activation
# from normalisation.logarithm import normalise  # , unnormalise
# from normalisation.meanvariance import normalise
from utils.crysollog import parseCrysolLogs, readDatsAndLogs, readLogs

import matplotlib
import pickle

# AGG backend is for writing to file, not for rendering in a window
matplotlib.use('Agg')
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

# folders = ["abs"]  # , "dat-c05", "dat-c1", "dat-c2", "dat-c4", "dat-c8", "dat-c16"]
folders = ["dat-c025", "dat-c05", "dat-c1", "dat-c2", "dat-c4", "dat-c8", "dat-c16"]

for f in folders:
    d = os.path.join(dataPath, "training", f)
    v = os.path.join(dataPath, "validation", f)
    fileNames = os.listdir(d)
    valNames = os.listdir(v)
    for ff in fileNames:
        dataFiles.append(os.path.join(d, ff))
    for ff in valNames:
        valFiles.append(os.path.join(v, ff))

t = os.path.join(dataPath, "test", f)
testNames = os.listdir(t)

if args.picklePath and (args.first != 1 or args.last):
    parser.error("If --picklePath is specified --first and --last will be ignored")

print("Reading data files...")

if not args.picklePath:
    # process --first and --last
    if args.first < 1:
        parser.error("--first must be at least = 1")
    firstPointIndex = args.first - 1
    cur, __ = saxsdocument.read(dataFiles[0])
    dat = cur['s']
    if args.last:
        if args.last > len(dat):
            parser.error(f"--last must be less or equal to the number of points in data files ({len(dat)})")
        lastPointIndex = args.last
    else:
        lastPointIndex = len(dat)

    smin = dat[firstPointIndex]
    smax = dat[lastPointIndex - 1]

    # read files
    Is, logFiles = readDatsAndLogs(dataFiles, logPath, firstPointIndex, lastPointIndex)
    IsVal, logFilesVal = readDatsAndLogs(valFiles, logPath, firstPointIndex, lastPointIndex)
    logFilesTest = readLogs(testNames, logPath)
    print("Parsing data log files...")
    parameters, outCsv = parseCrysolLogs(logFiles, par)
    maxValue = max(parameters)
    parameters = np.array(parameters) / maxValue
    print("...done.")

    print("Parsing validation log files...")
    parametersVal, outCsvVal = parseCrysolLogs(logFilesVal, par)
    parametersVal = np.array(parametersVal) / maxValue
    print("...done.")

    print("Parsing test log files...")
    parametersTest, outCsvTest = parseCrysolLogs(logFilesTest, par)
    print("...done.")

    # save to pickle
    pickle.dump([Is, logFiles, IsVal, logFilesVal, logFilesTest,
                 parameters, parametersVal, parametersTest,
                 firstPointIndex, lastPointIndex, smin, smax],
                open(f"data-{firstPointIndex}-{lastPointIndex}-{par}.p", "wb"))

    # save test set ground truth values to csv
    outCsvPath = f"ground-{par}-{len(logFilesTest)}.csv"
    np.savetxt(outCsvPath, outCsvTest, delimiter=",", fmt='%s')
    print(f"{outCsvPath} for test directory is written.")
else:
    Is, logFiles, IsVal, logFilesVal, logFilesTest, \
    parameters, parametersVal, parametersTest, \
    firstPointIndex, lastPointIndex, smin, smax = pickle.load(open(args.picklePath, "rb"))

print(f"Number of data files found: {len(dataFiles)}")
print(f"Number of log  files found: {len(logFiles)}")
print(f"Number of validation files found: {len(valFiles)}")
print(f"Number of validation log  files found: {len(logFilesVal)}")
print("...done.")



# Perceptron neural network
# tensorboard = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# Number of points in a SAXS curve
N = np.shape(Is)[1]

# Rg, Dmax, MW
output = np.shape(parameters)[1]

# Normalise SAXS input
# dd = np.ones(np.shape(Is)[1])  # no division
# Is, stdIs, meanIs = normalise(Is)
# IsVal, __, __ = normalise(IsVal, stdIs, meanIs)

# # DEBUG
# for I in IsVal[6:9]:
#     plt.plot(I)
# plt.savefig('validation-norm.png')
# plt.clf()
model = Sequential()
# first layer
# he = np.sqrt(0.06/N)
# model.add(Dense(args.units, input_dim=N, weights = [np.random.uniform(0,he,[args.units, N])]))
# model.add(Dense(args.units, input_dim=N, use_bias=True, kernel_initializer='glorot_uniform'))

model.add(Dense(args.units, input_dim=N, use_bias=False, kernel_initializer='he_uniform'))

# model.add(Dense(input_length, weights = [np.random.uniform(-he,he,[args.bottleneck_units, input_length]), averageIs]))
# model.add(Dense(input_length, weights = [np.zeros([args.hidden_units, input_length]), averageIs]))
model.add(Activation('tanh'))

# second layer
model.add(Dense(args.units, use_bias=False, kernel_initializer='he_uniform'))
model.add(Activation('tanh'))
# third layer
# model.add(Dense(args.units, use_bias=False, kernel_initializer='he_uniform'))
# model.add(Activation('tanh'))
# avrg = np.mean(parameters)
# print(f"Mean {par}: {avrg}")
# marginal improvement
# w = [np.zeros([args.units, 1]), np.array([avrg])]
model.add(Dense(output))  # , weights=w))
# model.add(Activation('relu'))

adama = optimizers.Adam(lr=0.005)  # , amsgrad=True, epsilon=0.1)  # lr=0.001 is default

model.compile(optimizer=adama, loss='mean_absolute_percentage_error')  # was loss='mse'

model_name = f"gnnom-{par}-{firstPointIndex}-{lastPointIndex}-e{args.epochs}-u{args.units}"

if args.weightsPath:
    model.load_weights(args.weightsPath)

# Check there are no Nans after normalisation
if np.isnan(Is).any():
    print("Error: Is matrix contains Nans!")
    quit()
if np.isnan(IsVal).any():
    print("Error: IsVal matrix contains Nans")
    quit()

train_history = model.fit(np.array(Is), np.array(parameters), epochs=num_epochs, batch_size=len(dataFiles),
                          validation_data=(np.array(IsVal), np.array(parametersVal)),
                          callbacks=[ModelCheckpoint(model_name + '.h5', save_best_only=True)])

loss = train_history.history['loss']
val_loss = train_history.history['val_loss']

# Confirm that it works
data = np.arange(N)

# save a 2d plot of the weights of the first layer
# plt.imshow(model.get_weights()[0], cmap='coolwarm')
# plt.savefig('weights-' + model_name + '.png')
# plt.clf()

# save the weights of the first layer
step = (smax - smin) / (lastPointIndex - firstPointIndex - 1)
s = np.arange(smin, smax + step, step)
np.savetxt(f'weights-{model_name}.int', np.column_stack((s, model.get_weights()[0])), fmt="%.8e")

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
model_json['outputNormalization'] = maxValue
try:
    model_json['meanIs'] = list(meanIs)
    model_json['stdIs'] = list(stdIs)
except:
    print("No normalization has been applied.")
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
