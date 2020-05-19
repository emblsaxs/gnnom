#!/usr/bin/python
# coding: utf-8

import argparse

parser = argparse.ArgumentParser(description='Make NN model - arguments and options.')
parser.add_argument('dataPath',    metavar='data',   type=str, help='path to the training data folder')
parser.add_argument('logPath',     metavar='logs',   type=str, help='path to the training log folder')
parser.add_argument('epochs',      metavar='epochs', type=int, help='number of epochs')
parser.add_argument('parameter',   metavar='parameter', type=str, help='mw/dmax/rg')
parser.add_argument('--units', type=int,   default=40, help='number of units in the hidden layer (default: 40)')
parser.add_argument('--first', type=int,   default=1,  help='index of the first point to use (default: 1)')
parser.add_argument('--last',  type=int,   default=-1, help='index of the last point to use (default: use all)')
parser.add_argument('--weightsPath', '-w', default=None, type=str, help='path to the h5 file')

args = parser.parse_args()


import keras
import numpy as np
import saxsdocument
import os
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import losses, optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

#np.random.seed(5)
#tf.random.set_seed(5)


num_epochs = int(args.epochs)
par = args.parameter

dataFiles = os.listdir(args.dataPath)
logFiles = []#os.listdir(args.logPath)

#dataFiles.sort()
#logFiles.sort()

n_all   = len(dataFiles)
n_cases = int(n_all * 0.9)

print("Reading data files...")

Is = []


# Process --first and --last:
firstPointIndex = int(args.first) - 1

path = os.path.join(args.dataPath, dataFiles[0])
__, cur  = saxsdocument.read(path)
dat  = cur['I']
lastPointIndex = len(dat)

if(int(args.last) > lastPointIndex):
    print("--last must be less or equal to the number of points in data files: " + str(lastPointIndex))
    exit()

if(args.last != -1):
    lastPointIndex = int(args.last)

for file in dataFiles:
    path = os.path.join(args.dataPath, file)
    if os.path.isdir(path): continue
    logPath = os.path.join(args.logPath,file[:-3] + "log")
    if os.path.exists(logPath) == False: 
        dataFiles.remove(file)
        print(file)
        continue
    prop, cur  = saxsdocument.read(path)
    Is.append(cur['I'][firstPointIndex:lastPointIndex])
    logFiles.append(logPath)

averageIs = np.mean(Is, axis = 0)
#Is = Is - averageIs
print("Number of data files found: " + str(len(dataFiles)))
print("Number of log  files found: " + str(len(logFiles)))
print("...done.")
exit()
print("Parsing log files...")
parameters = []
outCsv     = []

for file in logFiles:
    lines = [line.strip() for line in open(file)]
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



# Perceptron neural network
#tensorboard = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

####
# Number of points in a SAXS curve
N = np.shape(Is)[1]

# Rg, Dmax, MW
output = np.shape(parameters)[1]


model = Sequential()
# first layer
#he = np.sqrt(0.06/N)
#model.add(Dense(args.units, input_dim=N, weights = [np.random.uniform(0,he,[args.units, N])]))
model.add(Dense(args.units, input_dim=N, use_bias=True, kernel_initializer='he_uniform'))
#model.add(Dense(input_length, weights = [np.random.uniform(-he,he,[args.bottleneck_units, input_length]), averageIs]))
#model.add(Dense(input_length, weights = [np.zeros([args.hidden_units, input_length]), averageIs]))
model.add(Activation('relu'))

# second layer
model.add(Dense(args.units, use_bias=True, kernel_initializer='he_uniform', bias_initializer='zeros'))
model.add(Activation('relu'))
# third layer
model.add(Dense(args.units, use_bias=True, kernel_initializer='he_uniform', bias_initializer='zeros'))
model.add(Activation('relu'))

avrgMW = np.mean(parameters[0:n_cases])
print(f"Mean {par}: {avrgMW}")
# marginal imporovement
w = [np.zeros([args.units, 1]), np.array([avrgMW])]
model.add(Dense(output, weights = w))
#model.add(Dense(output))
adama = optimizers.Adam(lr=0.0001)

model.compile(optimizer= adama, loss='mse')

model_name = f"gnnom-{par}-0-1-e{args.epochs}-u{args.units}"

if(args.weightsPath):
    model.load_weights(args.weightsPath)

train_history = model.fit(np.array(Is[0:n_cases]), np.array(parameters[0:n_cases]), epochs=num_epochs,  batch_size=n_all,
                          validation_data =  (np.array(Is[n_cases:n_all]), np.array(parameters[n_cases:n_all])),
                          callbacks = [ModelCheckpoint(model_name + '.h5', save_best_only=True)])


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

scores = model.evaluate(np.array(Is[0:n_cases]), np.array(parameters[0:n_cases]), verbose=0)

print(model.metrics_names)
print(scores)
 
#save ground true values to csv
outCsvPath = f"ground-{par}.csv"
np.savetxt(outCsvPath, outCsv, delimiter=",", fmt='%s')
print(outCsvPath + " is written.")
 
# serialize model to JSON
model_json = model.to_json()
#model_json['smin'] = smin
#model_json['smax'] = smax
#model_json['firstPointIndex'] = firstPointIndex  # including, starts from 0
#model_json['lastPointIndex']  = lastPointIndex   # excluding
#model_json['KratkyDegree']    = args.degree
with open(model_name + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights(model_name + ".h5") #last but not best weights
print("Saved model " + model_name + " to disk")

#print("average Rg over the learning set:   " + str(avrgRg))
