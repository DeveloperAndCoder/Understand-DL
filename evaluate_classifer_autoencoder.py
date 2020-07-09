import numpy as np
from keras.datasets import cifar10
from keras.models import load_model
import os
import sys
import collect_data
import argparse

import stl_model

import unet_model as um

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--autoencoder_path", required=True,
	help="Relative path to model", default="")
ap.add_argument("-c", "--classifier_path", required=True,
	help="Relative path to model", default="")
ap.add_argument('--testing', dest='test', action='store_true')
ap.add_argument('--training', dest='test', action='store_false')
ap.set_defaults(test=True)

ap.add_argument('--weight', dest='weight', action='store_true')
ap.add_argument('--no-weight', dest='weight', action='store_false')
ap.set_defaults(weight=False)

ap.add_argument("-d", "--dataset", required=True,
	help="Dataset name: cifar, stl")
args = vars(ap.parse_args())

print(args)
# modelarch = args["architecture"]

def make_array(y):
    a = [[0]*10 for i in range(y.shape[0])]
    for i in range(y.shape[0]):
        a[i][y[i][0]] = 1
    return np.asarray(a)

classifier = None
autoencoder = None
classifier = load_model(args["classifier_path"])
autoencoder = load_model(args["autoencoder_path"])

(x_train, y_train), (x_test, y_test) = ([], []), ([], [])
if args["dataset"] == "stl":
    (x_train, y_train), (x_test, y_test) = collect_data.STL10.load_data(collect_data.STL10()) #ratio in which to split train and test
elif args["dataset"] == "cifar":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
elif args["dataset"] == "imagenet":
    (x_train, y_train), (x_test, y_test) = collect_data.Imagenet.load_data(collect_data.Imagenet(), toResize=True, dims=(224,224))
else:
    exit(1)
print('Number of test samples:', y_test.shape[0])
x_train = x_train/255
x_test = x_test/255
y_train = make_array(y_train)
y_test = make_array(y_test)

#num_samples = 10
x = x_test#[:num_samples]
y = y_test
if args["test"] == False:
    x = x_train
    y = y_train
#y = model.predict(sample) #output of pretrained autoencoder

pred = classifier.predict(autoencoder.predict(x))  #predict output
mse_loss = (np.square(pred-y)).mean(axis=None)
correctly_predicted = 0

for i, j in zip(pred, y):
    max_pos_i = np.argmax(i)
    max_pos_j = np.argmax(j)
    if (max_pos_i == max_pos_j):
        correctly_predicted += 1

print('Correctly predicted', correctly_predicted)
print('Incorrectly Predicted', y.shape[0] - correctly_predicted)
print('Total', y.shape[0])
print("Testing dataset=" + str(args["test"]) + ' | Accuracy%', correctly_predicted * 100 / y.shape[0])
print('MSE Loss', mse_loss)
