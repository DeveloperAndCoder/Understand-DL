import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D, BatchNormalization
from keras.models import Sequential, load_model
from keras.utils import multi_gpu_model
from keras.datasets import cifar10
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import keras.backend as kb
import keras.optimizers
import matplotlib.pyplot as plt
from keras import Model
import os
from keras.callbacks import CSVLogger
import cv2
import collect_data
from tqdm import tqdm

csv_logger = CSVLogger('Log/log.csv', append=True, separator=';')
save_dir = 'saved_models/imagenet_2'
autoencoder = load_model(save_dir + '/autoencoder.h5')
# classifier = load_model(save_dir + '/classifier.h5')
# print(x_train.shape, y_train.shape)

def make_array(y, num_of_classes = 10):
    a = [[0]*num_of_classes for i in range(y.shape[0])]
    for i in range(y.shape[0]):
        a[i][y[i][0]] = 1
    return np.asarray(a)


# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
(x_train, y_train), (x_test, y_test) = collect_data.Imagenet.load_data(collect_data.Imagenet(), toResize=True, dims=(224,224))
x_train = x_train / 255
x_test = x_test / 255
y_test = make_array(y_test, 1000)
y_train = make_array(y_train, 1000)

def form_collage(nplistA, nplistB, filepath):
    if nplistA.size != nplistB.size:
        print('Cannot save images: list a and b are of different size, ', nplistA.shape, nplistB.shape)
        return
    t = tqdm(total = len(nplistA))
    for i in range(0, len(nplistA)):
        predimg = nplistA[i]
        testimg = nplistB[i]
        testimg *= 255
        predimg *= 255
        concat = np.concatenate((testimg, predimg), axis=1)
        cv2.imwrite(filepath + str(i) + '.png', concat)
        t.update(1)

def test_autoencoder():
    pred_imgs = autoencoder.predict(x_test)
    # plt.imshow(sbscompare(x_test, pred_imgs, 20, 20))
    plt.axis('off')
    #plt.rcParams["figure.figsize"] = [60, 60]
    form_collage(pred_imgs, x_test, 'AutoEncoderResults/result')

def test_combined():
    combined = load_model(save_dir + '/w_combined.h5')
    print('Predicting...')
    # for image, label in zip(x_test, y_test):
    feat_classifier = classifier.predict(x_train)
    feat_combined = combined.predict(x_train)
    # print(pred.shape, y_test.shape)
    print('Prediction complete')

    x_wrong = []  # np.empty((0, x_test.shape[1], x_test.shape[2], x_test.shape[3]))
    y_wrong = []  # np.empty((0, y_test.shape[1]))
    # x_wrong = np.empty(x_test.shape[1:])

    autoencoder_output = []
    pred_img_autoencoder = autoencoder.predict(x_test)

    for i in range(0, len(x_test)):
        # feat = feats[i]
        # train_img = np.asarray(x_train[i]).reshape((-1, x_train[i].shape[0], x_train[i].shape[1], x_train[i].shape[2]))
        # label = y_train[i]
        pred_class_classifer = np.argmax(feat_classifier[i])
        pred_class_combined = np.argmax(feat_combined[i])
        true_class = np.argmax(y_test[i])
        # print(label, label.shape, label.shape[0])
        # print(pred_class, label, true_class)
        if pred_class_classifer != true_class and pred_class_combined == true_class:
            # print(true_class, pred_class)
            # print(x_wrong.shape, train_img.shape)
            x_wrong.append(x_test[i])  # = np.append(train_img, x_wrong, axis = 0)
            autoencoder_output.append(pred_img_autoencoder[i])
            # label = np.asarray(label).reshape((-1, label.shape[0]))
            # print(label.shape, y_wrong.shape)
            y_wrong.append(y_test[i])  # = np.append(label, y_wrong, axis = 0)

    x_wrong = np.asarray(x_wrong)
    y_wrong = np.asarray(y_wrong)
    autoencoder_output = np.asarray(autoencoder_output)

    form_collage(x_wrong, autoencoder_output, 'Images/Collage/')

    print("Final shape = ", x_wrong.shape, y_wrong.shape)
    print("Original shape = ", x_test.shape, y_test.shape)
    print("Accuracy = ", 100 - ((x_wrong.shape[0] / x_test.shape[0]) * 100))

def test_classifier():
    # combined = load_model(save_dir + '/f_combined.h5')
    print('Predicting...')
    # for image, label in zip(x_test, y_test):
    feats = classifier.predict(x_train)
    # print(pred.shape, y_test.shape)
    print('Prediction complete')

    x_wrong = []  # np.empty((0, x_test.shape[1], x_test.shape[2], x_test.shape[3]))
    y_wrong = []  # np.empty((0, y_test.shape[1]))
    # x_wrong = np.empty(x_test.shape[1:])

    x_wrong = np.asarray(x_wrong)
    y_wrong = np.asarray(y_wrong)

    for i in range(0, len(x_test)):
        # feat = feats[i]
        # train_img = np.asarray(x_train[i]).reshape((-1, x_train[i].shape[0], x_train[i].shape[1], x_train[i].shape[2]))
        # label = y_train[i]
        pred_class = np.argmax(feats[i])
        true_class = np.argmax(y_test[i])
        # print(label, label.shape, label.shape[0])
        # print(pred_class, label, true_class)
        if pred_class != true_class:
            # print(true_class, pred_class)
            # print(x_wrong.shape, train_img.shape)
            x_wrong.append(x_test[i])  # = np.append(train_img, x_wrong, axis = 0)
            # label = np.asarray(label).reshape((-1, label.shape[0]))
            # print(label.shape, y_wrong.shape)
            y_wrong.append(y_test[i])  # = np.append(label, y_wrong, axis = 0)

    print("Final shape = ", x_wrong.shape, y_wrong.shape)
    print("Original shape = ", x_test.shape, y_test.shape)

    form_collage(x_wrong, x_test, 'Images/Collage/')


def test_Vgg16():
    global x_train
    model = VGG16()
    print('Predicting...')
    x_train = x_train[:100]
    x_train = preprocess_input(x_train)
    print('x train shape = ', x_train.shape)
    # for image, label in zip(x_test, y_test):
    feats = model.predict(x_test)
    # feats = decode_predictions(feats)
    # print(pred.shape, y_test.shape)
    print('Prediction complete')
    # print(feats[0])
    # exit(0)
    x_wrong = []  # np.empty((0, x_test.shape[1], x_test.shape[2], x_test.shape[3]))
    y_wrong = []  # np.empty((0, y_test.shape[1]))
    # x_wrong = np.empty(x_test.shape[1:])

    # x_wrong = np.asarray(x_wrong)
    # y_wrong = np.asarray(y_wrong)

    for i in range(0, len(x_test)):
        # feat = feats[i]
        # train_img = np.asarray(x_train[i]).reshape((-1, x_train[i].shape[0], x_train[i].shape[1], x_train[i].shape[2]))
        # label = y_train[i]
        pred_class = np.argmax(feats[i])
        true_class = np.argmax(y_test[i])
        # print(label, label.shape, label.shape[0])
        print(pred_class, true_class)
        if pred_class != true_class:
            # print(true_class, pred_class)
            # print(x_wrong.shape, train_img.shape)
            x_wrong.append(x_test[i])  # = np.append(train_img, x_wrong, axis = 0)
            # label = np.asarray(label).reshape((-1, label.shape[0]))
            # print(label.shape, y_wrong.shape)
            y_wrong.append(y_test[i])  # = np.append(label, y_wrong, axis = 0)

    x_wrong = np.asarray(x_wrong)
    y_wrong = np.asarray(y_wrong)

    print("Final shape = ", x_wrong.shape, y_wrong.shape)
    print("Original shape = ", x_test.shape, y_test.shape)
    print("Accuracy = ", 100 - ((x_wrong.shape[0] / x_test.shape[0]) * 100))


test_Vgg16()