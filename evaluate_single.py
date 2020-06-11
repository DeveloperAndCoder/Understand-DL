import numpy as np
from keras.datasets import cifar10
from keras.models import load_model
import os
import sys
import collect_data
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--runnum", required=True,
	help="Run number: eg stl10_4")
ap.add_argument("-m", "--model", required=True,
	help="Model name: classifier, combined, autoencoder, etc.")
ap.add_argument("-d", "--dataset", required=True,
	help="Dataset name: cifar, stl")
args = vars(ap.parse_args())

runnum = args["runnum"]
modelname = args["model"]
dataset = args["dataset"]

runnum.strip()
print("runnum:", runnum)

save_dir = "saved_models/{}/".format(runnum)

def make_array(y):
    a = [[0]*10 for i in range(y.shape[0])]
    for i in range(y.shape[0]):
        a[i][y[i][0]] = 1
    return np.asarray(a)


model = load_model(os.path.join(save_dir, modelname))

print('Loaded model', modelname)

(x_train, y_train), (x_test, y_test) = ([], []), ([], [])
if dataset == "stl":
    (x_train, y_train), (x_test, y_test) = collect_data.STL10.load_data(collect_data.STL10()) #ratio in which to split train and test
elif dataset == "cifar":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
elif dataset == "imagenet":
    (x_train, y_train), (x_test, y_test) = collect_data.Imagenet.load_data(collect_data.Imagenet(), toResize=True, dims=(224,224))
else:
    exit(1)
print('Number of test samples:', y_test.shape[0])
x_train = x_train/255
x_test = x_test/255
y_train = make_array(y_train)
y_test = make_array(y_test)

#num_samples = 10
sample = x_test#[:num_samples]
img_init = model.predict(sample) #output of pretrained autoencoder

pred = model.predict(x_test)  #predict output
mse_loss = (np.square(pred-y_test)).mean(axis=None)
correctly_predicted = 0

for i, j in zip(pred, y_test):
    max_pos_i = np.argmax(i)
    max_pos_j = np.argmax(j)
    if (max_pos_i == max_pos_j):
        correctly_predicted += 1

print('Correctly predicted', correctly_predicted)
print('Incorrectly Predicted', y_test.shape[0] - correctly_predicted)
print('Total', y_test.shape[0])
print('Accuracy%', correctly_predicted * 100 / y_test.shape[0])
print('MSE Loss', mse_loss)
