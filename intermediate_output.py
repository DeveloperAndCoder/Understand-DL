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
from PIL import Image
import os
from keras.callbacks import CSVLogger
import cv2
import collect_data
from tqdm import tqdm
from unet_model import custom_loss
import argparse
from pathlib import Path
from alive_progress import alive_bar

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--runnum", required=True,
	help="Run number: eg stl10_4")
args = vars(ap.parse_args())
print(args)

basedir = 'saved_models/' + args["runnum"] + "/combined"
autoencoder_dir = os.path.join(basedir, 'f_auto.h5')
classifier_dir = os.path.join(basedir, 'f_class.h5')

collage_width = {64:8, 128:16, 256:16, 512:32}

def make_array(y, num_of_classes = 10):
    a = [[0]*num_of_classes for i in range(y.shape[0])]
    for i in range(y.shape[0]):
        a[i][y[i][0]] = 1
    return np.asarray(a)

# block1_conv1
# block2_conv2
def visualize_classifier(x, total_imgs=990):
    classifier = load_model(classifier_dir)
    classifier.summary()
    layers = [layer for layer in classifier.layers][:-4]
    layers = layers[1:]
    layer_outputs = [layer.output for layer in layers]
    layer_names = [layer.name for layer in layers]
    # print('lo', layer_outputs)
    # print('na', layer_names)

    # Extracts the outputs of the top 12 layers
    # layer_name = "block1_conv2"
    # print('co', classifier.get_layer(layer_name).output)
    # activation_model = Model(inputs=classifier.input,
    #                                 outputs=classifier.get_layer(layer_name).output)  # Creates a model that will return these outputs, given the model input
    activation_model = Model(inputs=classifier.input, outputs=layer_outputs)
    activation_model.summary()
    outputs = activation_model.predict(x)
    first_layer_activation = outputs[0]
    print(first_layer_activation.shape, type(first_layer_activation))
    print(len(outputs), type(outputs))
    for i in range(0, len(layer_names)):
        layer_name = layer_names[i]
        output = outputs[i] # Output of each layer
        save_dir = 'Images/Intermediate/' + args["runnum"] + "/Classifier/" + layer_name
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with alive_bar(min(total_imgs, output.shape[0])) as bar:
            for j in range(0, min(total_imgs, output.shape[0])):     # For each image
                int_imgs = []
                for k in range(0, output.shape[3]): # For each channel
                    #name = '{:04d}_{:03d}.png'.format(j, k) # name - Image_Channel
                    #cv2.imwrite(os.path.join(save_dir, name), output[j, :, :, k]*255)
                    int_imgs.append(output[j, :, :, k]*255)
                split = np.array_split(int_imgs, collage_width[len(int_imgs)])
                cols = [np.vstack(x) for x in split]
                collage = np.hstack(cols)
                name = '{:04d}_collage.png'.format(j)
                #print(split.shape, len(cols), collage.shape)
                cv2.imwrite(os.path.join(save_dir, name), collage)
                # uncomment below line to make it save images in the same folder
                #cv2.imwrite(save_dir+'_'+name, collage)
                bar()

def visualize_combined(x, total_imgs=990):
    autoencoder = load_model(autoencoder_dir)
    x = autoencoder.predict(x)
    classifier = load_model(classifier_dir)
    classifier.summary()
    layers = [layer for layer in classifier.layers][:-4]
    layers = layers[1:]
    layer_outputs = [layer.output for layer in layers]
    layer_names = [layer.name for layer in layers]
    activation_model = Model(inputs=classifier.input, outputs=layer_outputs)
    activation_model.summary()
    outputs = activation_model.predict(x)
    first_layer_activation = outputs[0]
    print(first_layer_activation.shape, type(first_layer_activation))
    print(len(outputs), type(outputs))
    for i in range(0, len(layer_names)):
        layer_name = layer_names[i]
        output = outputs[i] # Output of each layer
        save_dir = 'Images/Intermediate/' + args["runnum"] + "/Combined/" + layer_name
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with alive_bar(min(total_imgs, output.shape[0])) as bar:
            for j in range(0, min(total_imgs, output.shape[0])):     # For each image
                int_imgs = []
                for k in range(0, output.shape[3]): # For each channel
                    #name = '{:04d}_{:03d}.png'.format(j, k) # name - Image_Channel
                    #cv2.imwrite(os.path.join(save_dir, name), output[j, :, :, k]*255)
                    int_imgs.append(output[j, :, :, k] * 255)
                split = np.array_split(int_imgs, collage_width[len(int_imgs)])
                cols = [np.vstack(x) for x in split]
                collage = np.hstack(cols)
                name = '{:04d}_collage.png'.format(j)
                # print(split.shape, len(cols), collage.shape)
                cv2.imwrite(os.path.join(save_dir, name), collage)
                # uncomment below line to make it save images in the same folder
                #cv2.imwrite(save_dir+'_'+name, collage)
                bar()

def visualize_diff(x, total_imgs=990):
    autoencoder = load_model(autoencoder_dir)
    x_pred = autoencoder.predict(x)
    classifier = load_model(classifier_dir)
    classifier.summary()
    layers = [layer for layer in classifier.layers][:-4]
    layers = layers[1:]
    layer_outputs = [layer.output for layer in layers]
    layer_names = [layer.name for layer in layers]
    activation_model = Model(inputs=classifier.input, outputs=layer_outputs)
    activation_model.summary()
    outputs = activation_model.predict(x)
    outputs_pred = activation_model.predict(x_pred)
    #first_layer_activation = outputs[0]
    #print(first_layer_activation.shape, type(first_layer_activation))
    #print(len(outputs), type(outputs))
    for i in range(0, len(layer_names)):
        layer_name = layer_names[i]
        output = outputs[i] # Output of each layer
        output_pred = outputs_pred[i]
        save_dir = 'Images/Intermediate/' + args["runnum"] + "/Diff/" + layer_name
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        Path('Images/Intermediate/' + args["runnum"] + "/Diff/Collage/").mkdir(parents=True, exist_ok=True)
        with alive_bar(min(total_imgs, output.shape[0])) as bar:
            for j in range(0, min(total_imgs, output.shape[0])):     # For each image
                diff_imgs = []
                jet_imgs = []
                for k in range(0, output.shape[3]): # For each channel
                    #name = '{:04d}_{:03d}.png'.format(j, k) # name - Image_Channel
                    #cv2.imwrite(os.path.join(save_dir, name), output[j, :, :, k]*255)
                    diff, jetcam = generate_diffmap(output[j, :, :, k], output_pred[j, :, :, k])
                    diff_imgs.append(diff)
                    jet_imgs.append(jetcam)
                split = np.array_split(diff_imgs, collage_width[len(diff_imgs)])
                cols = [np.vstack(x) for x in split]
                collage_diff = np.hstack(cols)
                split = np.array_split(jet_imgs, collage_width[len(jet_imgs)])
                cols = [np.vstack(x) for x in split]
                collage_jet = np.hstack(cols)
                collage = np.hstack([collage_diff, collage_jet])
                name = '{:04d}_collage.png'.format(j)
                # print(split.shape, len(cols), collage.shape)
                cv2.imwrite(os.path.join(save_dir, name), collage)
                #cv2.imwrite('Images/Intermediate/' + args["runnum"] + "/Diff/Collage/" + str(j) + "_" + layer_name, collage)
                # uncomment below line to make it save images in the same folder
                #cv2.imwrite(save_dir+'_'+name, collage)
                bar()

def generate_diffmap(img1, img2):
    diff = img1-img2
    diff = 255 - np.uint8(np.absolute(diff)*255)
    #diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff = np.uint8((diff-np.min(diff))/(np.max(diff)-np.min(diff)+1)*255)
    jetcam = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    #diff = np.asarray([diff]*3)
    #img = np.hstack([diff, jetcam])
    return diff, jetcam

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = collect_data.STL10.load_data(collect_data.STL10())
    x_train = x_train / 255
    x_test = x_test / 255
    y_test = make_array(y_test)
    y_train = make_array(y_train)
    # visualize_classifier(x_test, total_imgs=5)
    # visualize_combined(x_test, total_imgs=5)
    visualize_diff(x_test, total_imgs=5)
