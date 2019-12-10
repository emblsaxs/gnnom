#!/usr/bin/python
import keras
from keras.models import model_from_json
import json
import saxsdocument
import numpy as np
import sys
import math

# Call like this:
# python gnnom.py gnnom.json gnnom.h5 inputfile.dat 2.85 /(Rg)
json_filename  = sys.argv[1]
h5_filename    = sys.argv[2]
input_filename = sys.argv[3]
Rg = (float)(sys.argv[4])
input_length   = 113 # I(s) points
output_length  = 301 # p(r) points
NN_smin = 0.01176470 # 1/A
norm_coef = 1.0


try:
    doc  = saxsdocument.read(input_filename)
    dat  = np.transpose(np.array(doc.curve[0]))
    s  = dat[0]
    Is = dat[1]
except:
    print("Error: Could not read input data")

# sew missing head
if NN_smin < s[0]:
   print("Adding missing smallest angle points based on Rg = " + str(Rg))
   step = s[1] - s[0]
   # find number of missing points
   head_number = (int)(np.rint((s[0] - NN_smin )/step))
   ss = NN_smin
   Is_head = np.full(head_number, 0.0)
   #s_head  = np.full(head_number, 0.0)
   for i in range(head_number):
       #s_head[i] = ss
       Is_head[i] = math.exp(ss*ss*Rg*Rg/-3.0)
       ss += step

   #s = np.hstack((s_head, s))
   Is = np.hstack((Is_head, Is))


# Assume Smin is correct; fill up to required Smax with typical I(0.4) intensity (assuming I(0) = 1.0)
zeroes = np.full((input_length - len(Is)), Is[-1])
Is_extended = np.concatenate((Is, zeroes))

try:
    # load json and create model
    json_file = open(json_filename, 'r')
    loaded_model_json = json_file.read()
    json_data = json.loads(loaded_model_json)
    if('Normalization coefficient' in json_data):
        norm_coef = (float)(json_data['Normalization coefficient'])
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(h5_filename)
    print("Model loaded. Yeah!")

except Exception as e:
    print("Error: Oops, model can not be uploaded.")
    print(e)
    sys.exit(-1)

# Evaluate loaded model on test data
test = np.array([Is_extended, ])
pred = loaded_model.predict(test)



if len(pred[0]) == output_length:
    # Find Dmax: first negative point after max(p(r))
    max_pddf = np.argmax(pred)
    negIndex = np.argmax(pred[:,max_pddf:] < 0)
    # Crop p(r > Dmax), nullify last point
    pred = pred[:, 0: (negIndex + max_pddf + 1)]
    pred[:,-1] = 0.0

    r = np.arange(0.0, len(pred[0]), 1.0)
    print("Dmax [A]: " + str(r[-1]))
    pddf_predicted = np.vstack((r, norm_coef*pred))
    np.savetxt('pddf-predicted.dat', np.transpose(pddf_predicted), fmt = "%.8e")

#TODO: read type of model and output units, print here
elif len(pred[0]) < 10:
    for number in pred[0]:
        print(number)

