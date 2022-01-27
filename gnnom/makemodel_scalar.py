# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
parser.add_argument('--last', type=int, default=None, help='index of the last point to use (default: use all)')
parser.add_argument('--weightsPath', '-w', default=None, type=str, help='path to the h5 file')
parser.add_argument('--picklePath', '-p', default=None, type=str, help='path to the pickle file, by default data.p')
parser.add_argument('--mode', default="WARNING", type=str, help='Logging level (default = WARNING), DEBUG, INFO')

args = parser.parse_args()

import numpy as np
from mysaxsdocument import saxsdocument
import os
import json
from keras.callbacks import ModelCheckpoint  # , TensorBoard
from keras import optimizers  # , losses
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
from keras.regularizers import l2
# from normalisation.logarithm import normalise  # , unnormalise
from gnnom.normalisation.meanvariance import normalise
from gnnom.utils.crysollog import parseCrysolLogs, readDatsAndLogs, readLogs

import matplotlib
import pickle

# AGG backend is for writing to file, not for rendering in a window
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# todo: tf.random.set_seed(5)
import time
import logging
from utils.log import log_warning, log_and_raise_error, log_debug, log_info
logger = logging.getLogger(__name__)
if args.mode == 'DEBUG':
    logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True

start = time.monotonic()

num_epochs = args.epochs
par = args.parameter

# valPath       = args.valPath
dataPath = args.dataPath
logPath = args.logPath
dataFiles = []
valFiles = []

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
    log_warning(logger, "If --picklePath is specified --first and --last will be ignored")
log_debug(logger, "Reading data files...")

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
    log_debug(logger, "Parsing data log files...")
    parameters, outCsv = parseCrysolLogs(logFiles, par)
    log_debug(logger, "...done.")

    log_debug(logger, "Parsing validation log files...")
    parametersVal, outCsvVal = parseCrysolLogs(logFilesVal, par)
    log_debug(logger, "...done.")

    log_debug(logger, "Parsing test log files...")
    parametersTest, outCsvTest = parseCrysolLogs(logFilesTest, par)
    log_debug(logger, "...done.")

    # save to pickle
    pickle.dump([Is, logFiles, IsVal, logFilesVal, logFilesTest,
                 parameters, parametersVal, parametersTest,
                 firstPointIndex, lastPointIndex, smin, smax],
                open(f"data-{firstPointIndex}-{lastPointIndex}-{par}.p", "wb"))

    # save test set ground truth values to csv
    outCsvPath = f"ground-{par}-{len(logFilesTest)}.csv"
    np.savetxt(outCsvPath, outCsvTest, delimiter=",", fmt='%s')
    log_info(logger, f"{outCsvPath} for test directory is written.")
else:
    Is, logFiles, IsVal, logFilesVal, logFilesTest, \
    parameters, parametersVal, parametersTest, \
    firstPointIndex, lastPointIndex, smin, smax = pickle.load(open(args.picklePath, "rb"))

log_debug(logger, f"Number of data files found: {len(dataFiles)}")
log_debug(logger, f"Number of log  files found: {len(logFiles)}")
log_debug(logger, f"Number of validation files found: {len(valFiles)}")
log_debug(logger, f"Number of validation log  files found: {len(logFilesVal)}")
log_debug(logger, "...done.")

# Perceptron neural network
# tensorboard = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# Number of points in a SAXS curve
N = np.shape(Is)[1]

# Rg, Dmax, MW
output = np.shape(parameters)[1]

# Normalize SAXS input (on for Dmax)
if par == 'dmax':
    dd = np.ones(np.shape(Is)[1])  # no division
    Is, stdIs, meanIs = normalise(Is, dd)
    IsVal, __, __ = normalise(IsVal, dd, meanIs)

# Normalize Rg, Dmax, MW
multiplier = np.max(parameters)
parameters = np.divide(parameters, multiplier)
parametersVal = np.divide(parametersVal, multiplier)
parametersTest = np.divide(parametersTest, multiplier)

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

model.add(
    Dense(args.units, input_dim=N, use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.0)))

BatchNormalization(axis=1)

# model.add(Dense(input_length, weights = [np.random.uniform(-he,he,[args.bottleneck_units, input_length]), averageIs]))
# model.add(Dense(input_length, weights = [np.zeros([args.hidden_units, input_length]), averageIs]))
model.add(Activation('tanh'))

# second layer
model.add(Dense(args.units, use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.0)))
model.add(Activation('tanh'))
# third layer
model.add(Dense(args.units, use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.0)))
model.add(Activation('tanh'))
# avrg = np.mean(parameters)
# log_debug(logger, f"Mean {par}: {avrg}")
# marginal improvement
# w = [np.zeros([args.units, 1]), np.array([avrg])]
# model.add(Dense(output, weights=w))
# model.add(Activation('relu'))
model.add(Dense(output, use_bias=False))

adama = optimizers.Adam(lr=0.0001)  # , amsgrad=True, epsilon=0.1)  # lr=0.001 is default

model.compile(optimizer=adama, loss='mean_absolute_percentage_error')  # was loss='mse'

model_name = f"gnnom-{par}-{firstPointIndex}-{lastPointIndex}-e{args.epochs}-u{args.units}"

if args.weightsPath:
    model.load_weights(args.weightsPath)

# Check there are no Nans after normalisation
if np.isnan(Is).any():
    log_and_raise_error(logger, "Is matrix contains Nans!")
if np.isnan(IsVal).any():
    log_and_raise_error(logger, "IsVal matrix contains Nans")

train_history = model.fit(np.array(Is), np.array(parameters), epochs=num_epochs, batch_size=128,  # len(dataFiles),
                          validation_data=(np.array(IsVal), np.array(parametersVal)),
                          callbacks=[ModelCheckpoint(model_name + '.h5', save_best_only=True)])

loss = train_history.history['loss']
val_loss = train_history.history['val_loss']

# Confirm that it works
data = np.arange(N)

# save a 2d plot of the weights of the first layer
plt.imshow(model.get_weights()[1], cmap='coolwarm')
plt.savefig(f"weights-{model_name}-1.png")
log_info(logger, f"weights-{model_name}-1.png is saved.")
plt.clf()

# save a 2d plot of the weights of the second layer
# plt.imshow(model.get_weights()[2], cmap='coolwarm')
# plt.savefig('weights-' + model_name + '-2.png')
# plt.clf()

# save the weights of the first layer
step = (smax - smin) / (lastPointIndex - firstPointIndex - 1)
s = np.arange(smin, smax + step, step)
np.savetxt(f'weights-{model_name}-0.int', np.column_stack((s, model.get_weights()[0])), fmt="%.8e")
log_info(logger, f'weights-{model_name}-0.int is saved.')
# np.savetxt(f'weights-{model_name}-1.int', np.column_stack((s, np.transpose(model.get_weights()[1]) * model.get_weights()[0])), fmt="%.8e")

# save the loss history
np.savetxt(f'loss-{model_name}.int', np.transpose(np.vstack((np.arange(num_epochs), loss, val_loss))), fmt="%.8e")
log_info(logger, f'loss-{model_name}.int is saved.')
# model.summary()
# scores = model.evaluate(np.array(IsVal), np.array(parametersVal), verbose=0)
# print(f"Metrics: {model.metrics_names}")
# print(f"Scores: {scores}")

# serialize model to JSON
model_str = model.to_json()
model_json = json.loads(model_str)
model_json['smin'] = smin
model_json['smax'] = smax
model_json['firstPointIndex'] = firstPointIndex  # including, starts from 0
model_json['lastPointIndex'] = lastPointIndex  # excluding
try:
    model_json['meanIs'] = list(meanIs)
    model_json['stdIs'] = list(stdIs)
except:
    log_warning(logger, "No normalization has been applied.")
model_json['Normalization coefficient'] = multiplier
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
