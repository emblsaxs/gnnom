#!/usr/bin/python
# coding: utf-8

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
from normalisation.log import normalise  # , unnormalise

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

# np.random.seed(5)
# tf.random.set_seed(5)

import time

start = time.time()

num_epochs    = args.epochs
par           = args.parameter

#valPath       = args.valPath
dataPath      = args.dataPath
logPath      = args.logPath
dataFiles     = []
valFiles      = []
logFiles      = []
logFilesVal   = []

folders = ["dat-c025", "dat-c05", "dat-c1", "dat-c2", "dat-c4", "dat-c8", "dat-c16"]

for f in folders:
    d = os.path.join(dataPath, "training", f)
    v = os.path.join(dataPath, "validation", f)
    fileNames = os.listdir(d)
    valNames = os.listdir(v)
    for ff in fileNames: dataFiles.append(os.path.join(d, ff))
    for ff in valNames: valFiles.append(os.path.join(v, ff))
#logFiles.extend(os.listdir(args.logPath))

#dataFiles.sort()
#logFiles.sort()

# n_all   = len(dataFiles)
# n_cases = int(valFiles)

print("Reading data files...")

Is = []
IsVal = []

# process --first and --last
cur, __ = saxsdocument.read(dataFiles[0])
dat = cur['s']
if args.last:
    if (int(args.last) > len(dat)):
        print(f"--last must be less or equal to the number of points in data files: {args.last}")
        exit()
    lastPointIndex = int(args.last)
else:
    lastPointIndex = len(dat) - 1

firstPointIndex = int(args.first) - 1

smin = dat[firstPointIndex]
smax = dat[lastPointIndex]
# print(f"smin = {smin}")
# print(f"smax = {smax}")
# exit()
for file in dataFiles:
    name = os.path.basename(file)
    # path = os.path.join(args.dataPath, file)
    if os.path.isdir(file): continue
    log = name[:-4] + ".log"
    l = os.path.join(logPath, log)
    if os.path.exists(l) == False:
        dataFiles.remove(file)
        print(f"No logs: removed from training {file}")
        continue
    cur, prop = saxsdocument.read(file)
    Is.append(cur['I'][firstPointIndex:lastPointIndex])
    logFiles.append(l)

for file in valFiles:
    name = os.path.basename(file)
    #path = os.path.join(args.dataPath, file)
    if os.path.isdir(file): continue
    log = name[:-4] + ".log"
    l   = os.path.join(logPath,log)
    if os.path.exists(l) == False: 
        valFiles.remove(file)
        print(f"No logs: removed from validation {file}")
        continue
    cur, prop = saxsdocument.read(file)
    IsVal.append(cur['I'][firstPointIndex:lastPointIndex])
    logFilesVal.append(l)

# averageIs = np.mean(Is, axis = 0)
#Is = Is - averageIs
print(f"Number of data files found: {len(dataFiles)}")
print(f"Number of log  files found: {len(logFiles)}")
print(f"Number of validation files found: {len(valFiles)}")
print(f"Number of validation log  files found: {len(logFilesVal)}")
print("...done.")

print("Parsing data log files...")
parameters = []
outCsv     = []
for f in logFiles:
    #l    = file#os.path.join(args.logPath,file)
    file = os.path.basename(f)
    #print(f)
    lines = [line.strip() for line in open(f)]
    rgdmaxmw = []
    # Read 'Molecular Weight: 0.4330E+06':
    if par not in ["rg", "dmax", "mw"] : 
        print(f"Wrong parameter {par}! Please enter rg, dmax or mw")
    for line in lines:
        if par == "rg":
            if "slope" in line:
                rg = float(line.split()[-1])
                rgdmaxmw.append(rg)
                parameters.append(rgdmaxmw)
                outCsv.append(file[:-4] + ', ' + str(round(rg, 3)))
                break
        if par == "dmax":
            if "diameter" in line:
                dmax = float(line.split()[-1])
                rgdmaxmw.append(dmax)
                parameters.append(rgdmaxmw)
                outCsv.append(file[:-4] + ', ' + str(round(dmax, 3)))
                break         
        if par == "mw":
            if "Weight" in line:
                mw = float(line.split()[2])/1000.0
                #print(f"{file}: {mw} kDa")
                rgdmaxmw.append(mw)
                parameters.append(rgdmaxmw)
                outCsv.append(file[:-4] + ', ' + str(round(mw, 3)))
                break

print("...done.")

print("Parsing validation log files...")
parametersVal = []
for f in logFilesVal:
    #l = file#os.path.join(args.logPath,file)
    file = os.path.basename(f)
    lines = [line.strip() for line in open(f)]
    rgdmaxmwVal = []
    # Read 'Molecular Weight: 0.4330E+06':
    if par not in ["rg", "dmax", "mw"] : 
        print(f"Wrong parameter {par}! Please enter rg, dmax or mw")
    for line in lines:
        if par == "rg":
            if "slope" in line:
                rg = float(line.split()[-1])
                rgdmaxmwVal.append(rg)
                parametersVal.append(rgdmaxmwVal)
                outCsv.append(file[:-4] + ', ' + str(round(rg, 3)))
                break
        if par == "dmax":
            if "diameter" in line:
                dmax = float(line.split()[-1])
                rgdmaxmwVal.append(dmax)
                parametersVal.append(rgdmaxmwVal)
                outCsv.append(file[:-4] + ', ' + str(round(dmax, 3)))
                break         
        if par == "mw":
            if "Weight" in line:
                mw = float(line.split()[2])/1000.0
                #print(f"{file}: {mw} kDa")
                rgdmaxmwVal.append(mw)
                parametersVal.append(rgdmaxmwVal)
                outCsv.append(file[:-4] + ', ' + str(round(mw, 3)))
                break

print("...done.")
 
#save ground true values to csv
outCsvPath = f"ground-{par}-{len(logFiles)}.csv"
np.savetxt(outCsvPath, outCsv, delimiter=",", fmt='%s')
print(outCsvPath + " is written.")

# Perceptron neural network
# tensorboard = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

####
# Number of points in a SAXS curve
N = np.shape(Is)[1]

# Rg, Dmax, MW
output = np.shape(parameters)[1]

# Normalise SAXS input
Is, meanIs, stdIs = normalise(Is)
IsVal, __, __ = normalise(IsVal, meanIs, stdIs)

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

avrgMW = np.mean(parameters)
print(f"Mean {par}: {avrgMW}")
# marginal imporovement
w = [np.zeros([args.units, 1]), np.array([avrgMW])]
model.add(Dense(output, weights = w))
model.add(Activation('relu'))
#model.add(Dense(output))
adama = optimizers.Adam(lr=0.0001)

model.compile(optimizer= adama, loss='mse')

model_name = f"gnnom-{par}-0.1-5-e{args.epochs}-u{args.units}"

if(args.weightsPath):
    model.load_weights(args.weightsPath)

train_history = model.fit(np.array(Is), np.array(parameters), epochs=num_epochs,  batch_size=len(dataFiles),
                          validation_data =  (np.array(IsVal), np.array(parametersVal)),
                          callbacks = [ModelCheckpoint(model_name + '.h5', save_best_only=True)])
#train_history = model.fit(np.array(Is[0:n_cases]), np.array(parameters[0:n_cases]), epochs=num_epochs,  batch_size=n_all,
#                          validation_data =  (np.array(Is[n_cases:n_all]), np.array(parameters[n_cases:n_all])),
#                          callbacks = [ModelCheckpoint(model_name + '.h5', save_best_only=True)])

loss = train_history.history['loss']
val_loss = train_history.history['val_loss']
'''
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
'''
# Confirm that it works
data = np.arange(N)

plt.imshow(model.get_weights()[0], cmap='coolwarm')
plt.savefig('weights-' + model_name + '.png')
plt.clf()

np.savetxt('loss-' + model_name + '.int', np.transpose(np.vstack((np.arange(num_epochs),loss, val_loss))), fmt = "%.8e")

scores = model.evaluate(np.array(IsVal), np.array(parametersVal), verbose=0)

print(model.metrics_names)
print(scores)

# serialize model to JSON
model_str = model.to_json()
model_json = json.loads(model_str)
model_json['smin'] = smin
model_json['smax'] = smax
model_json['firstPointIndex'] = firstPointIndex  # including, starts from 0
model_json['lastPointIndex'] = lastPointIndex  # excluding
model_json['meanIs'] = meanIs
model_json['stdIs'] = stdIs
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

#print("average Rg over the learning set:   " + str(avrgRg))
