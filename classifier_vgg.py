'''
#Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

import keras
import keras.backend as K
import tensorflow as tf
from keras.layers import Dense
#import tensorflow.keras.preprocessing.image.image_dataset_from_directory
from keras.applications import VGG16
from keras.models import Sequential
from keras.callbacks import CSVLogger, ModelCheckpoint
import sys
import os
from pathlib import Path
import numpy as np
import collect_data

# from STL10 import stl_to_dataset

if(len(sys.argv) != 2) :
    print('There need to be only one argument - Run number given')
    exit(1)

runnum = str(sys.argv[1])
print("runnum = ", runnum)

save_dir = "saved_models/" + runnum + "/"
log_dir = "Log/" + runnum + "/classifier/"
checkpoint_dir = 'checkpoint/' + runnum + "/classifier/"

Path(save_dir).mkdir(parents=True, exist_ok=True)
Path(log_dir).mkdir(parents=True, exist_ok=True)
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

csv_logger = CSVLogger(log_dir + "classifier_log.csv", append=True, separator=';')
checkpoint_template = os.path.join(checkpoint_dir, "{epoch:03d}_{loss:.2f}.hdf5")
checkpoint = ModelCheckpoint(checkpoint_template, monitor='loss', save_weights_only=False, mode='auto', period=10, verbose=1)

batch_size = 32
num_of_classes = 10
num_epochs = 250
# save_dir = os.path.join(os.getcwd(), 'saved_models/' + runnum)
model_name = 'classifier.h5'


# The data, split between train and test sets:
#(x_train, y_train), (x_test, y_test) = stl_to_dataset.get_numpy(0.8)
(x_train, y_train), (x_test, y_test) = collect_data.STL10.load_data(collect_data.STL10())

#exit()
#print(np.max(y_train))

def make_array(y):
    a = [[0]*num_of_classes for i in range(y.shape[0])]
    for i in range(0,y.shape[0]):
        a[i][y[i][0]] = 1
    return np.asarray(a)

y_train = make_array(y_train)
y_test = make_array(y_test)
#y_train = keras.utils.to_categorical(y_train, num_of_classes)
#y_test = keras.utils.to_categorical(y_test, num_of_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print('y_train shape:', y_train.shape, 'y_test shape:', y_test.shape)
print('y_train shape:', y_train[0].shape, 'y_test shape[0]:', y_test[0].shape)
'''
train_ds = image_dataset_from_directory(
    directory = 'STL10/img/',
    labels = 'inferred',
    label_mode = 'categorical',
    batch_size = 32,
    image_size = (96,96))
'''


vgg16 = VGG16(
    include_top=False,
    pooling='max',
    input_shape = (96,96,3)
    )
model = Sequential()
model.add(vgg16)
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

#model = keras.models.Sequential()
#model.add(VGG16(include_top=False, input_shape=(96,96,3)))
#model.add(Flatten())
#model.add(Dense(num_of_classes))

model.summary()

def loss_func(y_true, y_pred):
    y_true = K.print_tensor(y_true, message='y_true = ')
    y_pred = K.print_tensor(y_pred, message='y_pred = ')
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss = cce(y_true, y_pred)
    return loss

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['categorical_accuracy'],
              # loss=loss_func
            )

model.fit(x_train, y_train, epochs=num_epochs, batch_size=32, callbacks=[csv_logger, checkpoint])
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
