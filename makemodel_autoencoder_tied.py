#!/usr/bin/python
# coding: utf-8

import argparse

parser = argparse.ArgumentParser(description='Make NN autoencoder model - arguments and options.')
parser.add_argument('dataPath', metavar='data', type=str, help='path to the noisy training data folder')
parser.add_argument('--smoothDataPath', '-s', default=None, type=str, help='path to the smooth training data folder')
parser.add_argument('--weightsPath', '-w', default=None, type=str, help='path to the h5 file')
parser.add_argument('--epochs', default=None,   type=int, help='number of epochs')
parser.add_argument('--bottleneck_units', '-bu', default=10,  type=int, help='number of units in bottleneck layer (default: 10)')
parser.add_argument('--hidden_units', '-hu',  default=None,   type=int, help='number of units in the two hidden layers (default: no hidden layers)')
parser.add_argument('--first',   type=int, default=1,    help='index of the first point to use (default: 1)')
parser.add_argument('--last',    type=int, default=None, help='index of the last point to use (default: use all)')
parser.add_argument('--degree',  type=float, default= 0.0, help='I = I*s^degree (default: 0)')
parser.add_argument('--valData',       type=str, help='path to the noisy validation data folder')
parser.add_argument('--smoothValData', type=str, help='path to the smooth validation data folder')
parser.add_argument('--batch_size', '-b',  type=int, default=None, help='batch size (default: all)')
parser.add_argument('-p', '--prefix', type=str, default='', help='prefix for the output file names')
parser.add_argument('--pickle', type=str, default=None, help='path to pickle files')

args = parser.parse_args()


import numpy as np
import saxsdocument# as saxsdocument
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
start = time.time()
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
#from tensorflow.keras import regularizers
import DenseTranspose

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def chi2Loss(y_true, y_pred):
    chi2 = K.square(y_true - y_pred)/ y_true
    return chi2

def readFiles(path, firstPointIndex = 0, lastPointIndex = None, degree = 0):
    # Make sure there are no subfolders!
    files = os.listdir(path)
    files.sort()
    m = "data"
    arr = []
    for f in files:
        p = os.path.join(path, f)
        #doc  = saxsdocument.read(p)
        #dat  = np.transpose(doc.curve[0])[1]
        __, cur = saxsdocument.read(p)
        Is = np.array(cur['I'])
        s  = np.array(cur['s'])
        #arr.append(dat[firstPointIndex:lastPointIndex])
        arr.append(Is[firstPointIndex:lastPointIndex]*s[firstPointIndex:lastPointIndex]**(degree))
    print(f"{path}: {len(arr)} {m} files have been read")
    return np.array(arr)


def readFilesOld(path, firstPointIndex = 0, lastPointIndex = None, degree = 0):
    Rg = 20.0
    # Make sure there are no subfolders!
    files = os.listdir(path)
    files.sort()
    arr = []
    for f in files:
        p = os.path.join(path, f)
        doc  = saxsdocument.read(p)
        s    = np.transpose(doc.curve[0])[0]
        Is   = np.transpose(doc.curve[0])[1]
        if s[0] != 0:
            # sew missing head
            step = s[1] - s[0]
            # find number of missing points
            head_number = (int)(np.rint((s[0]) / step))
            ss = 0.0
            s_head = np.full(head_number, 0.0)
            Is_head = np.full(head_number, 0.0)
            for i in range(head_number):
                s_head[i] = ss
                Is_head[i] = np.exp(ss * ss * Rg * Rg / -3.0)
                ss += step
            s = np.hstack((s_head, s))
            Is = np.hstack((Is_head, Is))
        arr.append(Is[firstPointIndex:lastPointIndex]*s[firstPointIndex:lastPointIndex]**(degree))
    return np.array(arr)


def predict_from_model(model, dataFolder):
    for inputFilename in os.listdir(dataFolder):
        __, cur = saxsdocument.read(os.path.join(dataFolder, inputFilename))
        Is = np.array(cur['I'][firstPointIndex:lastPointIndex])
        s  = np.array(cur['s'][firstPointIndex:lastPointIndex])
        test = np.array([Is, ])
        pred = model.predict(test)
        dat_decoded = pred[0]
        dat_decoded = np.vstack((s, dat_decoded))
        np.savetxt(inputFilename, np.transpose(dat_decoded), fmt="%.8e")

# Process --first and --last:
firstPointIndex = args.first - 1
lastPointIndex  = args.last

#folders = ["dat-c16", "dat-c025", "dat-c05", "dat-c1", "dat-c2", "dat-c4", "dat-c8"]
folders = ["dat-c8", "dat-c4"]
#folders = ["abs"]
if(not args.pickle):
    smoothDataFolder = os.path.join(args.dataPath, folders[0])
    if args.smoothDataPath:
        smoothDataFolder = args.smoothDataPath

    if args.valData:
        smoothValDataFolder = os.path.join(args.valData, folders[0])
        if args.smoothValData:
            smoothValDataFolder = args.smoothValData

    noisyIs  = []
    smoothIs = []

    # read training data
    for f in folders:
        d = os.path.join(args.dataPath, f)
        I = readFiles(d, firstPointIndex, lastPointIndex)
        noisyIs.extend(I)
        I = readFiles(smoothDataFolder)  # Read full length smoothData, do not crop
        smoothIs.extend(I)

    noisyIs  = np.array(noisyIs)
    smoothIs = np.array(smoothIs)
    n_all = len(noisyIs)

    # # determine smin and smax from the first noisy data file
    path = os.path.join(args.dataPath, folders[0])
    # filesData = os.listdir(path)
    # firstFile = os.path.join(path, filesData[0])
    # __, cur = saxsdocument.read(firstFile)
    # s = cur['s']
    #
    # smin = s[firstPointIndex]
    # if (args.last):
    #     smax = s[args.last - 1]
    # else:
    #     smax = s[-1]
    #     lastPointIndex = len(s)
    # print(f"smin: {smin}, smax: {smax}")


    # save the s axis from the first smooth data file
    filesData = os.listdir(path)
    firstFile = os.path.join(path, filesData[0])
    __, cur = saxsdocument.read(firstFile)
    s = cur['s']


    noisyIsVal  = []
    smoothIsVal = []

    # read validation data
    if args.valData:
        for f in folders:
            d = os.path.join(args.valData, f)
            I = readFiles(d, firstPointIndex, lastPointIndex)
            noisyIsVal.extend(I)
            I = readFiles(smoothValDataFolder)
            smoothIsVal.extend(I)
        noisyIsVal  = np.array(noisyIsVal)
        smoothIsVal = np.array(smoothIsVal)

    else:
        n_cases = int(n_all * 0.9)
        noisyIsVal  = noisyIs [n_cases:n_all]
        smoothIsVal = smoothIs[n_cases:n_all]
        noisyIs     = noisyIs [0:n_cases]
        smoothIs    = smoothIs[0:n_cases]

    # saving vars to files
    pickleFile = []
    pickleFile.append(s)
    pickleFile.append(noisyIs)
    pickleFile.append(smoothIs)
    pickleFile.append(noisyIsVal)
    pickleFile.append(smoothIsVal)
    pickle.dump(pickleFile, open("Is.p", 'wb'))
else:
    s, noisyIs, smoothIs, noisyIsVal, smoothIsVal = pickle.load(open("Is.p", "rb"))
    n_all = len(noisyIs)

# determine smin and smax
smin = s[firstPointIndex]
if (args.last):
    smax = s[args.last - 1]
else:
    smax = s[-1]
    lastPointIndex = len(s)
print(f"smin: {smin}, smax: {smax}")

# Data Normalization
# #divisor = np.max(smoothIs, axis = 0)
# #np.savetxt("maximum.dat", np.transpose(np.vstack((s, divisor))), fmt = "%.8e")
#
# divisor = np.percentile(smoothIs, 99, axis = 0)
# np.savetxt("percentile.dat", np.transpose(np.vstack((s, divisor))), fmt = "%.8e")
#
# np.savetxt("mean.dat",   np.transpose(np.vstack((s, np.mean(smoothIs, axis = 0)))),   fmt = "%.8e")
# np.savetxt("median.dat", np.transpose(np.vstack((s, np.median(smoothIs, axis = 0)))), fmt = "%.8e")
#
# ddd = np.vstack((s, smoothIs[0]))
# np.savetxt("first-curve-not-divided.dat", np.transpose(ddd), fmt = "%.8e")
#
# #minIs = np.min(smoothIs, axis = 0)
# #ddd = np.vstack((s, minIs))
# #np.savetxt("minimum.dat", np.transpose(ddd), fmt = "%.8e")
#
# smoothIs    = np.divide(smoothIs   , divisor)
# noisyIs     = np.divide(noisyIs    , divisor)
# smoothIsVal = np.divide(smoothIsVal, divisor)
# noisyIsVal  = np.divide(noisyIsVal , divisor)
#
# subtractor = np.zeros(len(smoothIs[0]))
# #subtractor = np.median(smoothIs, axis = 0)
#
# # smoothIs    = smoothIs    - subtractor
# # noisyIs     = noisyIs     - subtractor
# # smoothIsVal = smoothIsVal - subtractor
# # noisyIsVal  = noisyIsVal  - subtractor
#
# ddd = np.vstack((s, smoothIs[0]))
# np.savetxt("first-curve-divided.dat", np.transpose(ddd), fmt = "%.8e")



# Data Standardisation with AVERAGE
#meanIs = np.mean(smoothIs, axis = 0)
#ddd = np.vstack((s, meanIs))
#np.savetxt("MEEEEEAAAAAAAN.dat", np.transpose(ddd), fmt = "%.8e")
#
#divisor = np.std(smoothIs, axis = 0)
#
#smoothIs    = smoothIs    - meanIs
#noisyIs     = noisyIs     - meanIs
#smoothIsVal = smoothIsVal - meanIs
#noisyIsVal  = noisyIsVal  - meanIs
#
#smoothIs    = np.divide(smoothIs   , divisor)
#noisyIs     = np.divide(noisyIs    , divisor)
#smoothIsVal = np.divide(smoothIsVal, divisor)
#noisyIsVal  = np.divide(noisyIsVal , divisor)





# Data Standardisation with MEDIAN
# Extra division by mean:
# divisor = np.mean(smoothIs, axis = 0)
#
# smoothIs    = np.divide(smoothIs   , divisor)
# noisyIs     = np.divide(noisyIs    , divisor)
# smoothIsVal = np.divide(smoothIsVal, divisor)
# noisyIsVal  = np.divide(noisyIsVal , divisor)
#
# subtractor = np.median(smoothIs, axis = 0)
# ddd = np.vstack((s, subtractor))
# np.savetxt("median.dat", np.transpose(ddd), fmt = "%.8e")
#
# ddd = np.vstack((s, smoothIs[0]))
# np.savetxt("first-not-divided.dat", np.transpose(ddd), fmt = "%.8e")
#
# smoothIs    = smoothIs    - subtractor
# noisyIs     = noisyIs     - subtractor
# smoothIsVal = smoothIsVal - subtractor
# noisyIsVal  = noisyIsVal  - subtractor
#
# # a = divisor  # For the extra division by mean
# divisor = np.sqrt(np.sum(smoothIs**2, axis = 0) / len(smoothIs[0]))
#
# smoothIs    = np.divide(smoothIs   , divisor)
# noisyIs     = np.divide(noisyIs    , divisor)
# smoothIsVal = np.divide(smoothIsVal, divisor)
# noisyIsVal  = np.divide(noisyIsVal , divisor)
#
# # # Extra division by mean: to write correct values to json:
# # subtractor = np.divide(subtractor, divisor)
# # divisor    = np.multiply(a, divisor)
#
# ddd = np.vstack((s, divisor))
# np.savetxt("divisor.dat", np.transpose(ddd), fmt = "%.8e")
#
# ddd = np.vstack((s, smoothIs[0]))
# np.savetxt("first-divided.dat", np.transpose(ddd), fmt = "%.8e")
#

#stdIs = np.std(smoothIs, axis = 0)
#smoothIs    = np.divide(smoothIs,    stdIs)
#noisyIs     = np.divide(noisyIs,     stdIs)
#smoothIsVal = np.divide(smoothIsVal, stdIs)
#noisyIsVal  = np.divide(noisyIsVal,  stdIs)

medianIs  = np.median(smoothIs, axis = 0)

ddd = np.vstack((s, medianIs))
np.savetxt("divided-median.dat", np.transpose(ddd), fmt = "%.8e")



if args.epochs:
    num_epochs = int(args.epochs)
else:
    num_epochs = int(20000000.0 / len(noisyIs))


#tensorboard = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# Number of points in a SAXS curve
input_length  = np.shape(noisyIs)[1]
output_length = np.shape(smoothIs)[1]


model = Sequential()

# output layer
#he = np.sqrt(0.06/input_length)
#model.add(Dense(input_length, weights = [np.random.uniform(-he,he,[args.bottleneck_units, input_length]), averageIs]))

if args.hidden_units:
    # three layers
    model.add(Dense(2 * args.hidden_units, input_dim=input_length, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('tanh'))
    
    model.add(Dense(args.hidden_units, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('tanh'))
    
    model.add(Dense(args.bottleneck_units, use_bias=False, kernel_initializer='zeros', bias_initializer='zeros'))
    model.add(Activation('tanh'))
    
    model.add(Dense(args.hidden_units, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('tanh'))
    
    model.add(Dense(2 * args.hidden_units, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('tanh'))
    
    #model.add(Dense(output_length, weights = [np.zeros([2 * args.hidden_units, output_length]), medianIs]))
    model.add(Dense(output_length, bias_initializer='zeros', use_bias=False))
else:
    # # single layer
    # model.add(Dense(args.bottleneck_units, input_dim=input_length, use_bias=False, kernel_initializer='zeros', bias_initializer='zeros'))
    # # model.add(Dense(args.bottleneck_units, input_dim=input_length, use_bias=True, kernel_initializer='he_uniform', bias_initializer='zeros', activity_regularizer=regularizers.l1(10e-7)))
    # model.add(Activation('tanh'))
    #
    # # model.add(Dense(output_length, weights = [np.zeros([args.bottleneck_units, output_length]), medianIs]))
    # model.add(Dense(output_length, bias_initializer='zeros', use_bias=False))

    # Tied weights
    inputs = keras.Input(shape=input_length)
    dense_1 = Dense(128, activation='tanh')
    dense_2 = Dense(64, activation='tanh')
    dense_3 = Dense(args.bottleneck_units, activation='tanh')
    x = dense_1(inputs)
    x = dense_2(x)
    x = dense_3(x)
    x = DenseTranspose.DenseTranspose(dense_3, activation = "tanh")(x)
    x = DenseTranspose.DenseTranspose(dense_2, activation = "tanh")(x)
    x = DenseTranspose.DenseTranspose(dense_1, activation = "tanh")(x)
    outputs = Dense(input_length)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)


# output layer
#he = np.sqrt(0.06/input_length)
#model.add(Dense(input_length, weights = [np.random.uniform(-he,he,[args.bottleneck_units, input_length]), averageIs]))

#model.add(Dense(input_length, weights = [np.zeros([args.bottleneck_units, input_length]), averageIs]))

#model.add(Dense(input_length))

#for layer in range(args.layers - 1):
#   # add layer
#   model.add(Dense(args.units, use_bias=True, kernel_initializer='he_uniform', bias_initializer='zeros'))
#   model.add(Activation('relu'))




#o = Adam(lr=1e-3)#, epsilon = 1e-8, beta_1 = .9, beta_2 = .999)
#o = 'sgd'

o = 'Adam'
#o = optimizers.Adam(learning_rate = 0.0002)

if(args.weightsPath):
    model.load_weights(args.weightsPath)
    o = optimizers.Adam(learning_rate = 0.00002)
l = 'huber_loss'  # doesn't fit high angles?..
#l = losses.mean_absolute_error  # so far best
#l = losses.mean_squared_error  # doesn't fit low angles
#l = losses.mean_absolute_percentage_error  # poor fit at all angles
#l = losses.mean_squared_logarithmic_error  # doesn't fit low angles AT ALL
#l = losses.cosine_similarity # garbage
#l = losses.LogCosh() # about the same as mean_absolute_error
#l = chi2Loss

model.compile(optimizer=o, loss=l)

model_name = f"autoencoder-{args.prefix}-e{num_epochs}-bu{args.bottleneck_units}-l1-d{args.degree}"

if(not args.batch_size):
    bs = n_all
else:
    bs = args.batch_size

train_history = model.fit(noisyIs, smoothIs, epochs=num_epochs,  batch_size=bs,
                          validation_data =  (noisyIsVal, smoothIsVal))#,callbacks=[ModelCheckpoint(model_name + '.h5', save_best_only=True)])
path = "dat-c8"
predict_from_model(model, path)

model.save_weights(model_name + '.h5')

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


#data = np.arange(input_length)

# Plot weights
plt.imshow(model.get_weights()[0], cmap='coolwarm')
plt.savefig('weights-' + model_name + '.png')
plt.clf()

# Save loss as .int file
np.savetxt('loss-' + model_name + '.int', np.transpose(np.vstack((np.arange(num_epochs),loss, val_loss))), fmt = "%.8e")

# FIXME:
# Apply the last (not best) model to the first file in the validation set
#pred = model.predict(np.array([noisyIsVal[0], ]))
#dat_decoded = np.vstack((s, pred[0]))
#np.savetxt('pred-' + model_name + '.dat', np.transpose(dat_decoded), fmt = "%.8e")

# compute consumed time
end = time.time()
t   = str(round((end - start) / 60,2))

# serialize model to JSON
model_str = model.to_json()
model_json = json.loads(model_str)
model_json['output-s-axis'] = s
# model_json['subtractor'] = (list)(subtractor)
# #model_json['stdIs']  = (list)(stdIs)
# #model_json['minIs'] = (list)(minIs)
# model_json['divisor'] = (list)(divisor)
model_json['smin'] = smin
model_json['smax'] = smax
model_json['firstPointIndex'] = firstPointIndex  # including, starts from 0
model_json['lastPointIndex']  = lastPointIndex   # excluding
model_json['KratkyDegree']    = args.degree
model_json['minutesTrained']    = t
print(f"Time consumed: {t} minutes")

with open(model_name + ".json", "w") as json_file:
    json_file.write(json.dumps(model_json))
# serialize weights to HDF5
#model.save_weights(model_name + ".h5") #last but not best weights
print(f"Saved model {model_name} to disk")
