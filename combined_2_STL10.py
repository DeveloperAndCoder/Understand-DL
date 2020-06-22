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
import argparse

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, "0" to  "7"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--runnum", required=True,
	help="Run number: eg stl10_4")
ap.add_argument("-a", "--autoencoder", required=False,
	help="Relative path to autoencoder", default="")
ap.add_argument("-c", "--classifier", required=False,
	help="Relative path to classifier", default="")
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# ap.add_argument('--unet', dest='unet', action='store_true')
# ap.add_argument('--no-unet', dest='unet', action='store_false')
# ap.set_defaults(unet=False)

args = vars(ap.parse_args())

print(args)

=======

args = vars(ap.parse_args())

>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======

args = vars(ap.parse_args())

>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======

args = vars(ap.parse_args())

>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======

args = vars(ap.parse_args())

>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
runnum = args["runnum"]
autoencoder_path = args["autoencoder"]
classifier_path = args["classifier"]
# dataset = args["dataset"]

runnum.strip()
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
print("runnum:", runnum)
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
print("runnum:", runnum)
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
print("runnum:", runnum)
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
print("runnum:", runnum)
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377

save_dir = "saved_models/" + runnum + "/"
log_dir = "Log/" + runnum + "/combined/"
checkpoint_dir = 'checkpoint/' + runnum + "/combined/"

Path(save_dir + "combined/").mkdir(parents=True, exist_ok=True)
Path(log_dir).mkdir(parents=True, exist_ok=True)
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

csv_logger = CSVLogger(log_dir + "combined_log.csv", append=True, separator=';')
checkpoint_template = os.path.join(checkpoint_dir, "{epoch:03d}_{loss:.2f}.hdf5")
checkpoint = ModelCheckpoint(checkpoint_template, monitor='loss', save_weights_only=False, mode='auto', period=10, verbose=1)

autoencoder_dir = save_dir

if autoencoder_path == "":
    autoencoder_path = autoencoder_dir + 'autoencoder.h5'

#autoencoder = load_model('saved_models/autoencoder.h5')
autoencoder = load_model(autoencoder_path)
# classifier model= load_model(save_dir+'/classifier.h5')
num_of_classes = 10

vgg = True

if classifier_path == "":
    classifier_path = save_dir + 'classifier.h5'

classifier = load_model(classifier_path)
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
(x_train, y_train), (x_test, y_test) = collect_data.STL10.load_data(collect_data.STL10(), train_perc = 80)
=======
(x_train, y_train), (x_test, y_test) = collect_data.STL10.load_data(collect_data.STL10())
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
(x_train, y_train), (x_test, y_test) = collect_data.STL10.load_data(collect_data.STL10())
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
(x_train, y_train), (x_test, y_test) = collect_data.STL10.load_data(collect_data.STL10())
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
(x_train, y_train), (x_test, y_test) = collect_data.STL10.load_data(collect_data.STL10())
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
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

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
num_epochs=250
adam = keras.optimizers.Adam(learning_rate=1e-5)

>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
num_epochs=250
adam = keras.optimizers.Adam(learning_rate=1e-5)

>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
num_epochs=250
adam = keras.optimizers.Adam(learning_rate=1e-5)

>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
num_epochs=250
adam = keras.optimizers.Adam(learning_rate=1e-5)

>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
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

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
adam = keras.optimizers.Adam(learning_rate=1e-4)

=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
if vgg and False:
    print('using vgg loss')
    combined.compile(optimizer=adam, metrics=['accuracy'], loss=vgg_loss)
else :
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    print('Using categorical cross entropy')
    combined.compile(optimizer=adam, metrics=['categorical_accuracy'], loss='categorical_crossentropy')
print("Compiled!!!!")

combined.fit(x_train, y_train, epochs=250, callbacks=[csv_logger, checkpoint])
# combined.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=250, callbacks=[csv_logger, checkpoint])
=======
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
    print('Using mean sq loss')
    combined.compile(optimizer=adam, metrics=['accuracy'], loss='mean_squared_error')
print("Compiled!!!!")

combined.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=num_epochs, callbacks=[csv_logger, checkpoint])
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
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
