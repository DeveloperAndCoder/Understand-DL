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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
from PIL import Image
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
import os
from keras.callbacks import CSVLogger
import cv2
import collect_data
from tqdm import tqdm
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
from unet_model import custom_loss
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--runnum", required=True,
	help="Run number: eg stl10_4")
ap.add_argument("-a", "--autoencoder", required=False,
	help="Relative path to autoencoder", default="")
ap.add_argument("-c", "--classifier", required=False,
	help="Relative path to classifier", default="")
# ap.add_argument('--unet', dest='unet', action='store_true')
# ap.add_argument('--no-unet', dest='unet', action='store_false')
# ap.set_defaults(unet=False)

args = vars(ap.parse_args())
print(args)

runnum = args["runnum"]
autoencoder_path = args["autoencoder"]
classifier_path = args["classifier"]

runnum.strip()


csv_logger = CSVLogger('Log/log.csv', append=True, separator=';')
save_dir = 'saved_models/' + runnum
if autoencoder_path == "":
    autoencoder_path = save_dir + '/combined/f_auto.h5'

# unet_auto = load_model(save_dir+'/unet.h5', custom_objects = {'custom_loss': custom_loss})
#unet_auto = load_model(save_dir+'/unet.h5')
#exit()
if classifier_path == "":
    classifier_path = save_dir + '/combined/f_class.h5'

# print(x_train.shape, y_train.shape)
class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
for c in class_names:
    try:
        # os.mkdir('Images/'+c)
        Path('Images/' + runnum + '/' + c).mkdir(parents=True, exist_ok=True)
    except:
        pass
=======
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377

csv_logger = CSVLogger('Log/log.csv', append=True, separator=';')
save_dir = 'saved_models/imagenet_2'
autoencoder = load_model(save_dir + '/autoencoder.h5')
# classifier = load_model(save_dir + '/classifier.h5')
# print(x_train.shape, y_train.shape)
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

def make_array(y, num_of_classes = 10):
    a = [[0]*num_of_classes for i in range(y.shape[0])]
    for i in range(y.shape[0]):
        a[i][y[i][0]] = 1
    return np.asarray(a)


# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# (x_train, y_train), (x_test, y_test) = collect_data.Imagenet.load_data(collect_data.Imagenet(), toResize=True, dims=(224,224))
(x_train, y_train), (x_test, y_test) = collect_data.STL10.load_data(collect_data.STL10())
x_train = x_train / 255
x_test = x_test / 255
y_test = make_array(y_test)
y_train = make_array(y_train)

def form_collage(nplistA, nplistB, sdir):
    print('making collage', nplistA.shape)
    if nplistA.size != nplistB.size:
        print('Cannot save images: list a and b are of different size, ', nplistA.shape, nplistB.shape)
        return
    #t = tqdm(total = len(nplistA))
    for i in range(0, nplistA.shape[0]):
=======
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
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
        predimg = nplistA[i]
        testimg = nplistB[i]
        testimg *= 255
        predimg *= 255
        concat = np.concatenate((testimg, predimg), axis=1)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        #print(concat.shape)
        name =  '{:04d}.png'.format(i)
        #img = Image.fromarray(concat)
        print('Saving image', os.path.join(sdir, name))
        cv2.imwrite(os.path.join(sdir, name), concat)
        #print('saved', name)
        #t.update(1)
    return

def test_autoencoder(filepath = 'AutoEncoderResults/'):
    autoencoder = load_model(autoencoder_path)
    save_dir = os.path.join(filepath, runnum)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    pred_imgs = unet_auto.predict(x_test)
    # plt.imshow(sbscompare(x_test, pred_imgs, 20, 20))
    plt.axis('off')
    #plt.rcParams["figure.figsize"] = [60, 60]
    form_collage(pred_imgs, x_test, save_dir)

def test_combined():
    autoencoder = load_model(autoencoder_path)
    classifier = load_model(classifier_path)
    combined = load_model(save_dir + '/combined/f_combined.h5')
    print('Predicting...')
    # for image, label in zip(x_test, y_test):
    feat_classifier = classifier.predict(x_test)
    feat_combined = combined.predict(x_test)
=======
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
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
    # print(pred.shape, y_test.shape)
    print('Prediction complete')

    x_wrong = []  # np.empty((0, x_test.shape[1], x_test.shape[2], x_test.shape[3]))
    y_wrong = []  # np.empty((0, y_test.shape[1]))
    # x_wrong = np.empty(x_test.shape[1:])

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    orig_wrong = 0
    comb_wrong = 0

    autoencoder_output = []
    pred_img_autoencoder = autoencoder.predict(x_test)

    class_wrong_comb_right_img = []
    class_wrong_comb_right_auto = []

=======
    autoencoder_output = []
    pred_img_autoencoder = autoencoder.predict(x_test)

>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
    autoencoder_output = []
    pred_img_autoencoder = autoencoder.predict(x_test)

>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
    autoencoder_output = []
    pred_img_autoencoder = autoencoder.predict(x_test)

>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
    autoencoder_output = []
    pred_img_autoencoder = autoencoder.predict(x_test)

>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
    for i in range(0, len(x_test)):
        # feat = feats[i]
        # train_img = np.asarray(x_train[i]).reshape((-1, x_train[i].shape[0], x_train[i].shape[1], x_train[i].shape[2]))
        # label = y_train[i]
        pred_class_classifer = np.argmax(feat_classifier[i])
        pred_class_combined = np.argmax(feat_combined[i])
        true_class = np.argmax(y_test[i])
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        #cv2.imwrite('Images/{}/{:03d}.png'.format(class_names[pred_class_combined], i), x_test[i]*255)
        #print('{}\t{}\t{}'.format(pred_class_classifer, pred_class_combined, true_class))
        # print(label, label.shape, label.shape[0])
        # print(pred_class, label, true_class)
        if pred_class_classifer != true_class:
            orig_wrong+=1
            # print('Images/' + runnum + '/{}/{:03d}.png'.format(class_names[pred_class_classifer], i))
            cv2.imwrite('Images/' + runnum + '/{}/{:03d}.png'.format(class_names[pred_class_classifer], i), x_test[i]*255)
        if pred_class_combined != true_class:
            comb_wrong+=1
            x_wrong.append(x_test[i])
            autoencoder_output.append(pred_img_autoencoder[i])
            y_wrong.append(y_test[i])

        if pred_class_classifer != true_class and pred_class_combined == true_class:
            # print(true_class, pred_class)
            # print(x_wrong.shape, train_img.shape)
            class_wrong_comb_right_img.append(x_test[i])  # = np.append(train_img, x_wrong, axis = 0)
            class_wrong_comb_right_auto.append(pred_img_autoencoder[i])
            # label = np.asarray(label).reshape((-1, label.shape[0]))
            # print(label.shape, y_wrong.shape)
            # y_wrong.append(y_test[i])  # = np.append(label, y_wrong, axis = 0)
=======
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
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

    x_wrong = np.asarray(x_wrong)
    y_wrong = np.asarray(y_wrong)
    autoencoder_output = np.asarray(autoencoder_output)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

    Path('Images/'+runnum+'/Collage/').mkdir(parents=True, exist_ok=True)
    form_collage(x_wrong, autoencoder_output, 'Images/'+runnum+'/Collage/')

    class_wrong_comb_right_img = np.asarray(class_wrong_comb_right_img)
    class_wrong_comb_right_auto = np.asarray(class_wrong_comb_right_auto)
    Path('Images/' + runnum + '/Collage_of_correct_combined_incorrect_classifier/').mkdir(parents=True, exist_ok=True)
    form_collage(class_wrong_comb_right_img, class_wrong_comb_right_auto, 'Images/' + runnum + '/Collage_of_correct_combined_incorrect_classifier/')

    print("Wrongly classified = ", x_wrong.shape, y_wrong.shape)
    print("Original shape = ", x_test.shape, y_test.shape)
    print(orig_wrong, comb_wrong)
    print("Accuracy = ", 100 - ((x_wrong.shape[0] / x_test.shape[0]) * 100))

def test_classifier():
    classifier = load_model(classifier_path)
    # combined = load_model(save_dir + '/f_combined.h5')
    print('Predicting...')
    # for image, label in zip(x_test, y_test):
    feats = classifier.predict(x_test)
=======
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
    form_collage(x_wrong, autoencoder_output, 'Images/Collage/')

    print("Final shape = ", x_wrong.shape, y_wrong.shape)
    print("Original shape = ", x_test.shape, y_test.shape)
    print("Accuracy = ", 100 - ((x_wrong.shape[0] / x_test.shape[0]) * 100))

def test_classifier():
    # combined = load_model(save_dir + '/f_combined.h5')
    print('Predicting...')
    # for image, label in zip(x_test, y_test):
    feats = classifier.predict(x_train)
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
    # print(pred.shape, y_test.shape)
    print('Prediction complete')

    x_wrong = []  # np.empty((0, x_test.shape[1], x_test.shape[2], x_test.shape[3]))
    y_wrong = []  # np.empty((0, y_test.shape[1]))
    # x_wrong = np.empty(x_test.shape[1:])

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
    x_wrong = np.asarray(x_wrong)
    y_wrong = np.asarray(y_wrong)

>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
    x_wrong = np.asarray(x_wrong)
    y_wrong = np.asarray(y_wrong)

>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
    x_wrong = np.asarray(x_wrong)
    y_wrong = np.asarray(y_wrong)

>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
    x_wrong = np.asarray(x_wrong)
    y_wrong = np.asarray(y_wrong)

>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
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

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    x_wrong = np.asarray(x_wrong)
    y_wrong = np.asarray(y_wrong)

    print("Final shape = ", x_wrong.shape, y_wrong.shape)
    print("Original shape = ", x_test.shape, y_test.shape)

    # form_collage(x_wrong, x_test, 'Images/' + runnum + '/Collage/')

def wrongly_classified(x_test, y_test):
    path = save_dir+'/after_classifier.h5'
    classifier = load_model(path)
    y_pred = classifier.predict(x_test)
    x_wrong = []
    for i in range(0, len(x_test)):
        pred_class = np.argmax(y_pred[i])
        true_class = np.argmax(y_test[i])
        if pred_class != true_class:
            x_wrong.append(x_test[i])
            y_wrong.append(y_test[i])
    return x_wrong
=======
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
    print("Final shape = ", x_wrong.shape, y_wrong.shape)
    print("Original shape = ", x_test.shape, y_test.shape)

    form_collage(x_wrong, x_test, 'Images/Collage/')

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


<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# test_Vgg16()
#test_autoencoder()
test_combined()
# test_classifier()
=======
test_Vgg16()
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
test_Vgg16()
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
test_Vgg16()
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
test_Vgg16()
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
