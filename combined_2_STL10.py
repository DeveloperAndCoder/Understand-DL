import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D, BatchNormalization, Dense
from keras.models import Sequential, load_model
from keras.utils import multi_gpu_model
from keras.datasets import cifar10
import keras.backend as kb
import keras.optimizers
import matplotlib.pyplot as plt
from keras import Model
import os
from keras.callbacks import CSVLogger, ModelCheckpoint
import collect_data
from pathlib import Path
import sys
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

if(len(sys.argv) != 2) :
    print('There need to be only one argument - Run number given')
    exit(1)

runnum = str(sys.argv[1])
runnum.strip()
print("runnum:", runnum)

save_dir = "saved_models/" + runnum + "/"
log_dir = "Log/" + runnum + "/combined/"
checkpoint_dir = 'checkpoint/' + runnum + "/combined/"

Path(save_dir + "combined/").mkdir(parents=True, exist_ok=True)
Path(log_dir).mkdir(parents=True, exist_ok=True)
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

csv_logger = CSVLogger(log_dir + "combined_log.csv", append=True, separator=';')
checkpoint_template = os.path.join(checkpoint_dir, "{epoch:03d}_{loss:.2f}.hdf5")
checkpoint = ModelCheckpoint(checkpoint_template, monitor='loss', save_weights_only=True, mode='auto', period=1, verbose=1)

autoencoder_dir = save_dir

#autoencoder = load_model('saved_models/autoencoder.h5')
autoencoder = load_model(autoencoder_dir + 'autoencoder.h5')
# classifier model= load_model(save_dir+'/classifier.h5')
num_of_classes = 10

vgg = True
classifier = load_model(save_dir + 'classifier.h5')
# vgg16 = VGG16(
#     include_top=False,
#     pooling='max',
#     input_shape = (96,96,3)
#     )
# classifier = Sequential()
# classifier.add(vgg16)
# classifier.add(Dense(64, activation='relu'))
# classifier.add(Dense(10, activation='softmax'))
# classifier.summary()
#classifier = Model(vgg16.inputs, classifier(vgg16.outputs))
# exit()
(x_train, y_train), (x_test, y_test) = collect_data.STL10.load_data(collect_data.STL10())
#(x_train, y_train), (x_test, y_test) = get_numpy()
#(x_train, y_train), (x_test, y_test) = collect_data.Imagenet.load_data(collect_data.Imagenet(), toResize=True, dims=(224,224))
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
### cifar - (50k, 32, 32, 3) (50k, 1) (10k, 32, 32, 3) (10k, 1)
x_train = x_train/255
x_test = x_test/255

#exit(1)


def make_array(y):
    a = [[0]*num_of_classes for i in range(y.shape[0])]
    for i in range(0,y.shape[0]):
        a[i][y[i][0]] = 1
    return np.asarray(a)

print(type(x_train), type(y_train), type(x_test), type(y_test))
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

y_test = make_array(y_test)
y_train = make_array(y_train)

print(type(x_train), type(y_train), type(x_test), type(y_test))
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

lambda_1 = 1
lambda_2 = 1

def makeUntrainable(layer):
    layer.trainable = False
    if hasattr(layer, 'layer'):
      for l in layer.layers:
        makeUntrainable(l)

makeUntrainable(classifier)

inputs = autoencoder.inputs
autoencoder.summary()
print('autoencoder.outputs =', autoencoder.outputs)
outputs = classifier(autoencoder.outputs)
combined = Model(inputs, outputs)
# combined.add(Dense(64, activation='relu'))
# combined.add(Dense(10, activation='softmax'))
combined.summary()

num_epochs=1000
adam = keras.optimizers.Adam(learning_rate=1e-5)

def get_map():
    synsets = open('synsets.txt', 'r')
    wnids = []
    labels = []
    for line in synsets:
        w, l = line.split()
        wnids.append(w)
        labels.append(l)
    map_from_foldername_to_wid = {}
    for i in range(0, len(wnids)):
        map_from_foldername_to_wid.setdefault(labels[i], wnids[i])
    return map_from_foldername_to_wid

def vgg_loss(y_true, y_pred):
    print(type(y_true), y_true.shape)
    print(y_true)
    # y_true = decode_predictions(y_true)
    # print(len(y_true))
    tf.print(y_true)
    # print(y_true[0][0])
    # exit(1)
    return (y_true - y_pred)

if vgg and False:
    print('using vgg loss')
    combined.compile(optimizer=adam, metrics=['accuracy'], loss=vgg_loss)
else :
    print('Using mean sq loss')
    combined.compile(optimizer=adam, metrics=['accuracy'], loss='mean_squared_error')
print("Compiled!!!!")

combined.fit(x_train, y_train, epochs=num_epochs, callbacks=[csv_logger, checkpoint])
save_dir = os.path.join(os.getcwd(), save_dir + "combined/")

model_name = 'f_auto.h5'
model_path = os.path.join(save_dir, model_name)
autoencoder.save(model_path)

model_name = 'f_class.h5'
model_path = os.path.join(save_dir, model_name)
classifier.save(model_path)

model_name = 'f_combined.h5'
model_path = os.path.join(save_dir, model_name)
combined.save(model_path)
