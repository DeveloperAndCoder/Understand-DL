import sys
import os
import glob
import numpy as np
import cv2
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.models import Sequential, load_model
import tensorflow as tf
from tensorflow.python.framework import ops
import argparse
import collect_data
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--runnum", required=True,
	help="Run number: eg stl10_4")

args = vars(ap.parse_args())

print(args)

def make_array(y):
    a = [[0]*10 for i in range(y.shape[0])]
    for i in range(y.shape[0]):
        a[i][y[i][0]] = 1
    return np.asarray(a)

#(x_train, y_train), (x_test, y_test) = collect_data.STL10.load_data(collect_data.STL10(), train_perc = 80)
(x_train, y_train), (x_test, y_test) = collect_data.Intel.load_data(collect_data.Intel())
x_train = x_train/255
x_test = x_test/255
y_train = make_array(y_train)
y_test = make_array(y_test)

num_samples = 1
x=x_test#[:num_samples]
y=y_test#[:num_samples]
#x = x*255
root = 'Images/Diff/'+args['runnum']
Path(root).mkdir(parents=True, exist_ok=True)

autoencoder = load_model("saved_models/" + args["runnum"] + "/combined/f_auto.h5")
x_pred = autoencoder.predict(x)

for i in range(x.shape[0]):
    diff = x_pred[i]-x[i]
    diff = np.uint8(np.absolute(diff)*255)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff = np.uint8((diff-np.min(diff))/(np.max(diff)-np.min(diff))*255)
    jetcam = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    #diff = np.asarray([diff]*3)
    img = np.hstack([x[i]*255, x_pred[i]*255, diff, jetcam])
    cv2.imwrite(root+'/{:04d}.png'.format(i), img)

print('done')
    
