import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D, BatchNormalization
from keras.models import Sequential, load_model
from keras.utils import multi_gpu_model
from keras.datasets import cifar10
import keras.backend as kb
import keras.optimizers
import matplotlib.pyplot as plt
from keras import Model
import os
from keras.callbacks import CSVLogger

csv_logger = CSVLogger('Log/log.csv', append=True, separator=';')

save_dir = 'saved_models'

autoencoder = load_model(save_dir+'/autoencoder.h5')
classifier = load_model(save_dir+'/classifier.h5')
#combined = load_model(save_dir + '/f_combined.h5')

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train/255
x_test = x_test/255

print(x_train.shape, y_train.shape)

def make_array(y):
    a = [[0]*10 for i in range(y.shape[0])]
    for i in range(y.shape[0]):
        a[i][y[i][0]] = 1
    return np.asarray(a)

y_test = make_array(y_test)
y_train = make_array(y_train)

lambda_1 = 1
lambda_2 = 1

print('Predicting...')
#for image, label in zip(x_test, y_test):
feats = classifier.predict(x_train)
#print(pred.shape, y_test.shape)
print('Prediction complete')


x_wrong = []#np.empty((0, x_test.shape[1], x_test.shape[2], x_test.shape[3]))
y_wrong = []#np.empty((0, y_test.shape[1]))
#x_wrong = np.empty(x_test.shape[1:])


for i in range(0, len(x_train)):    
    #feat = feats[i]
    #train_img = np.asarray(x_train[i]).reshape((-1, x_train[i].shape[0], x_train[i].shape[1], x_train[i].shape[2]))
    #label = y_train[i]
    pred_class = np.argmax(feats[i])
    true_class = np.argmax(y_train[i])
    #print(label, label.shape, label.shape[0])
    #print(pred_class, label, true_class)
    if pred_class != true_class:
        #print(true_class, pred_class)
        #np.vstack((x_wrong, feat))
        #print(x_wrong.shape, train_img.shape)
        x_wrong.append(x_train[i])# = np.append(train_img, x_wrong, axis = 0)
        #label = np.asarray(label).reshape((-1, label.shape[0]))
        #print(label.shape, y_wrong.shape)
        y_wrong.append(y_train[i])# = np.append(label, y_wrong, axis = 0)


x_wrong = np.asarray(x_wrong)
y_wrong = np.asarray(y_wrong)
print("Final shape = ", x_wrong.shape, y_wrong.shape)
#exit(0)

def makeUntrainable(layer):
    layer.trainable = False
    if hasattr(layer, 'layer'):
      for l in layer.layers:
        makeUntrainable(l)

makeUntrainable(classifier)

inputs = autoencoder.inputs
outputs = classifier(autoencoder.outputs)
combined = Model(inputs, outputs)

num_epochs=1000
adam = keras.optimizers.Adam(learning_rate=1e-5)
combined.compile(optimizer=adam, metrics=['accuracy'], loss='mean_squared_error')
print("Compiled!!!!")
combined.fit(x_wrong, y_wrong, epochs=num_epochs, batch_size=128, callbacks=[csv_logger])
save_dir = os.path.join(os.getcwd(), 'saved_models')

model_name = 'w_auto.h5'
model_path = os.path.join(save_dir, model_name)
autoencoder.save(model_path)

model_name = 'w_class.h5'
model_path = os.path.join(save_dir, model_name)
classifier.save(model_path)

model_name = 'w_combined.h5'
model_path = os.path.join(save_dir, model_name)
combined.save(model_path)
