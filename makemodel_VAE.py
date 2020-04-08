#!/usr/bin/python
# coding: utf-8

import argparse

parser = argparse.ArgumentParser(description='Make NN autoencoder model - arguments and options.')
parser.add_argument('dataPath', metavar='data', type=str, help='path to the training data folder')
parser.add_argument('--weightsPath', '-w', default=None, type=str, help='path to the h5 file')
parser.add_argument('--epochs', default=None,   type=int, help='number of epochs')
parser.add_argument('--bottleneck_units', '-bu', default=10,  type=int, help='number of units in bottleneck layer (default: 10)')
parser.add_argument('--hidden_units', '-hu',  default=80,     type=int, help='number of units in the hidden layer (default: 80)')
parser.add_argument('--first',   type=int, default= 1, help='index of the first point to use (default: 1)')
parser.add_argument('--last',    type=int, default=-1, help='index of the last point to use (default: use all)')
parser.add_argument('--degree',  type=float, default= 0.0, help='I = I*s^degree (default: 0)')
parser.add_argument('--valData', type=str, help='path to the validation data folder')
parser.add_argument('-p', '--prefix', type=str, default='', help='prefix for the output file names')

args = parser.parse_args()


import numpy as np
import saxsdocument
import os
import json
import random
#To force using CPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Lambda, Input
from keras import losses, optimizers
from keras import backend as K
from keras.utils import plot_model

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def readFiles(path, firstPointIndex = 0, lastPointIndex = None, degree = 0):
    # Make sure there are no subfolders!
    files = os.listdir(path)
    files.sort()
    random.shuffle(files)
    arr = []
    for f in files:
        p = os.path.join(path, f)
        if os.path.isdir(p):
            # skip directories
            continue
        prop, cur  = saxsdocument.read(p)
        Is   = cur['I']
        arr.append(Is)
    return np.array(arr)

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

#another hyperparameter to adjust
LOSS_FACTOR = 50000
def r_loss(y_true, y_pred):
    return LOSS_FACTOR*K.mean(K.square(y_true - y_pred))

def kl_loss(mean_mu, log_var):
    kl_loss =  -0.5 * K.sum(1 + log_var - K.square(mean_mu) - K.exp(log_var))
    return kl_loss


def total_loss(y_true, y_pred):
    return r_loss(y_true, y_pred) + kl_loss(y_true, y_pred)


# Process --first and --last:
firstPointIndex = int(args.first) - 1

# Read first I(s) file to get number of points
file = os.listdir(args.dataPath)[0]
path = os.path.join(args.dataPath, file)
prop, cur  = saxsdocument.read(path)
dat  = cur['s']

lastPointIndex = len(dat)
if(int(args.last) > lastPointIndex):
    print(f"--last must be less or equal to the number of points in data files: {lastPointIndex}")
    exit()
if(args.last != -1):
    lastPointIndex = int(args.last)

smin = dat[firstPointIndex]
smax = dat[lastPointIndex - 1]

#read training set files
Is   = readFiles(args.dataPath, firstPointIndex, lastPointIndex, args.degree)
n_all = len(Is)
averageIs = np.mean(Is, axis = 0)

#create validation set
if args.valData:
    IsVal   = readFiles(args.valData, firstPointIndex, lastPointIndex, args.degree)

else:
    n_cases = int(n_all * 0.9)
    IsVal   =   Is[n_cases:n_all]
    Is      =   Is[0:n_cases]

if args.epochs:
    num_epochs = int(args.epochs)
else:
    num_epochs = int(20000000.0 / len(Is))


#tensorboard = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# Number of points in a SAXS curve
input_length  = np.shape(Is)[1]
# network parameters
input_shape = (input_length, )
intermediate_dim = args.hidden_units
latent_dim = args.bottleneck_units

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(input_length, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')


# output layer
#he = np.sqrt(0.06/input_length)
#model.add(Dense(input_length, weights = [np.random.uniform(-he,he,[args.bottleneck_units, input_length]), averageIs]))
#model.add(Dense(input_length, weights = [np.zeros([args.hidden_units, input_length]), averageIs]))

if(args.weightsPath):
    model.load_weights(args.weightsPath)
#adama = optimizers.Adam(learning_rate=0.0001)
#adam = Adam(lr=1e-3)#, epsilon = 1e-8, beta_1 = .9, beta_2 = .999)

vae.compile(optimizer='Adam', loss=total_loss, metrics=[r_loss, kl_loss])#losses.huber_loss)

model_name = f"autoencoder-{args.prefix}-e{num_epochs}-bu{args.bottleneck_units}-l3-d{args.degree}"

train_history = vae.fit(Is, Is, epochs=num_epochs,  batch_size=len(Is),
                          validation_data =  (IsVal, IsVal),
                          callbacks = [ModelCheckpoint(model_name + '.h5', save_best_only=True)])


loss = train_history.history['loss']
val_loss = train_history.history['val_loss']

data = np.arange(input_length)

plt.imshow(vae.get_weights()[0], cmap='coolwarm')
plt.savefig('weights-' + model_name + '.png')
plt.clf()

np.savetxt('loss-' + model_name + '.int', np.transpose(np.vstack((np.arange(num_epochs),loss, val_loss))), fmt = "%.8e")

 
# serialize model to JSON
model_str = vae.to_json()
model_json = json.loads(model_str)
model_json['smin'] = smin
model_json['smax'] = smax
model_json['firstPointIndex'] = firstPointIndex  # including, starts from 0
model_json['lastPointIndex']  = lastPointIndex   # excluding
model_json['KratkyDegree']    = args.degree

with open(model_name + ".json", "w") as json_file:
    json_file.write(json.dumps(model_json))
# serialize weights to HDF5
#model.save_weights(model_name + ".h5") #last but not best weights
print(f"Saved model {model_name} to disk")

