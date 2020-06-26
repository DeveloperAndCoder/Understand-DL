from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
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


def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    img_path = path
    img = image.load_img(img_path, target_size=(224, 224))
    #img = image.load_img(img_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='block5_conv3'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        # new_model = VGG16(weights='imagenet')
        new_model = load_model(args["model"])
    return new_model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var)
    for var, grad in zip(var_list, grads)]

def grad_cam(input_model, image, category_index, layer_name):
    model = Sequential()
    model.add(input_model)
    model.summary()
    nb_classes = 1000
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    model.add(Lambda(target_layer,
                     output_shape = target_category_loss_output_shape))

    print('model layers', model.layers, model.layers[-1].output) #output: model layers
    # [<keras.engine.training.Model object at 0x7f11642dcc50>,
    # <keras.layers.core.Lambda object at 0x7f11634cdf28>]
    # Tensor("lambda_1/Mul:0", shape=(?, 10), dtype=float32)
    loss = K.sum(model.layers[-1].output)
    conv_output =  [l for l in model.layers[0].layers if l.name is layer_name][0].output
    # conv_output = [l for l in model.layers if l.name == layer_name][0].output
    #conv_output = model.get_layer('model_1').get_layer(layer_name).output
    print(conv_output) #output: Tensor("conv2d_23/Sigmoid:0", shape=(?, 96, 96, 3), dtype=float32)
    # exit(1)
    #print('fd', K.gradients(loss, conv_output), K.gradients(loss, conv_output)[0])
    #grads = normalize(K.gradients(loss, conv_output)[0])
    grads = normalize(_compute_gradients(loss, [conv_output])[0])
    #print('grads', grads, model.get_layer('model_1').input, keras.Input(shape=(96, 96, 3)))
    #gradient_function = K.function([model.layers[0].input], [conv_output, grads])
    print('blah', model.inputs[0], conv_output, grads)
    gradient_function = K.function([model.inputs[0]], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

def decode_predictions_stl(pred):
    cnums = np.argmax(pred, axis=1)
    cprob = np.max(pred, axis=1)
    return (cnums[0], cprob[0])

def makeUntrainable(layer):
    layer.trainable = False
    if hasattr(layer, 'layer'):
      for l in layer.layers:
        makeUntrainable(l)

print(args["image"])
preprocessed_input = load_image(args["image"])

model = VGG16(weights='imagenet')
#model = load_model(args["model"])
# makeUntrainable(model)

model.summary()

predictions = model.predict(preprocessed_input)
print('Predictions:', predictions)
top_1 = decode_predictions_stl(predictions)
print('Predicted class:', top_1)
# print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))

predicted_class = np.argmax(predictions)
# print(predicted_class, model, preprocess_input)
cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, "block5_conv3")
cv2.imwrite("gradcam.jpg", cam)

register_gradient()
guided_model = modify_backprop(model, 'GuidedBackProp')
saliency_fn = compile_saliency_function(guided_model)
saliency = saliency_fn([preprocessed_input, 0])
gradcam = saliency[0] * heatmap[..., np.newaxis]
cv2.imwrite("guided_gradcam.jpg", deprocess_image(gradcam))
