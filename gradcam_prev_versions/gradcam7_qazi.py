"""
https://keras.io/examples/vision/grad_cam/
Title: Grad-CAM class activation visualization
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/04/26
Last modified: 2020/05/14
Description: How to obtain a class activation heatmap for an image classification model.
"""
"""
Adapted from Deep Learning with Python (2017).

## Setup
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential, load_model
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
import argparse

ap = argparse.ArgumentParser()
# ap.add_argument("-r", "--runnum", required=True,
# 	help="Run number: eg stl10_4")
ap.add_argument("-m", "--model", required=True,
	help="Relative path to model")
ap.add_argument("-i", "--image", required=True,
	help="Relative path to image")
# ap.add_argument('--unet', dest='unet', action='store_true')
# ap.add_argument('--no-unet', dest='unet', action='store_false')
# ap.set_defaults(unet=False)

args = vars(ap.parse_args())

print(args)

model_path = args["model"]
img_path = args["image"]

# Display
#from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


"""
## Configurable parameters

You can change these to another model.

To get the values for `last_conv_layer_name` and `classifier_layer_names`, use
 `model.summary()` to see the names of all layers in the model.
"""

#model_builder = keras.applications.xception.Xception
img_size = (96, 96)
#preprocess_input = keras.applications.xception.preprocess_input
#decode_predictions = keras.applications.xception.decode_predictions

#last_conv_layer_name = "block5_conv3"
last_conv_layer_name = "block5_pool"
classifier_layer_names = [
    #"block5_pool",
    "flatten",
    "dense_1",
    "dropout_1",
    "dense_2",
]

# The local path to our target image
#img_path = keras.utils.get_file(
    #"african_elephant.jpg", " https://i.imgur.com/Bvro0YD.png"
#)

#display(Image(img_path))


"""
## The Grad-CAM algorithm
"""


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)
    
    classifier_model.summary()
    
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape(persistent=True) as tape:
        # Compute activations of the last conv layer and make the tape watch it
        #last_conv_layer_output = last_conv_layer_model(img_array)
        last_conv_layer_output = last_conv_layer_model(tf.convert_to_tensor(img_array, dtype=tf.float32))
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        print('last_conv_layer_output', last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0], output_type=tf.dtypes.int32)
        print('top_pred_index', top_pred_index, 'preds', preds)
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    print(tape)
    grads = tape.gradient(preds, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


"""
## Let's test-drive it
"""

# Prepare image
#img_array = preprocess_input(get_img_array(img_path, size=img_size))
img_array = get_img_array(img_path, size=img_size)
#img_array = cv2.imread(img_path)
#img_array = np.array(img_array)
#img_array = cv2.imread(img_path)
img_array = img_array / 255
#img_array = np.expand_dims(img_array, axis=0)

print('Image shape', img_array.shape, type(img_array))

# Make model
#model = model_builder(weights="imagenet")
model = load_model(model_path)

model.summary()

# Print what the top predicted class is
preds = model.predict(img_array)
#print("Predicted:", decode_predictions(preds, top=1)[0])

# Generate class activation heatmap
heatmap = make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
)

# Display heatmap
plt.matshow(heatmap)
plt.show()


"""
## Create a superimposed visualization
"""

# We load the original image
img = keras.preprocessing.image.load_img(img_path)
img = keras.preprocessing.image.img_to_array(img)

# We rescale heatmap to a range 0-255
heatmap = np.uint8(255 * heatmap)

# We use jet colormap to colorize heatmap
jet = cm.get_cmap("jet")

# We use RGB values of the colormap
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]

# We create an image with RGB colorized heatmap
jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

# Superimpose the heatmap on original image
superimposed_img = jet_heatmap * 0.4 + img
superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

# Save the superimposed image
save_path = "elephant_cam.jpg"
superimposed_img.save(save_path)

# Display Grad CAM
#display(Image(save_path)) 
