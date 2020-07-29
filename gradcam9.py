import sys
import os
import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras import backend as K
from keras.preprocessing import image
from keras.datasets import cifar10
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.models import Sequential, load_model
import tensorflow as tf
from tensorflow.python.framework import ops
import argparse
import collect_data
from pathlib import Path
from progress.bar import Bar
from alive_progress import alive_bar

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--runnum", required=True,
	help="Run number: eg stl10_4")
ap.add_argument('--combined', dest='combined', action='store_true')
ap.add_argument('--no-combined', dest='combined', action='store_false')
ap.set_defaults(combined=False)
ap.add_argument("-m", "--model", required=False,
	help="Relative path to model")
ap.add_argument("-i", "--image", required=False,
	help="Relative path to image")
# ap.add_argument('--unet', dest='unet', action='store_true')
# ap.add_argument('--no-unet', dest='unet', action='store_false')
# ap.set_defaults(unet=False)

args = vars(ap.parse_args())

print(args)

# Define model here ---------------------------------------------------
def build_model():
    """Function returning keras model instance.
    
    Model can be
     - Trained here
     - Loaded with load_model
     - Loaded from keras.applications
    """
    if args["model"]:
        return load_model(args["model"])
    else:
        return load_model("saved_models/" + args["runnum"] + "/combined/f_class.h5")
    #return VGG16(include_top=True, weights='imagenet')

H, W = 144, 144 # Input shape, defined by the model (model.input_shape)
# ---------------------------------------------------------------------

def load_image(path, preprocess=True):
    """Load and preprocess image."""
    #x = image.load_img(path, target_size=(H, W))
    x=path
    if preprocess:
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        #x = preprocess_input(x)
    return x


def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)


def build_guided_model():
    """Function returning modified model.
    
    Changes gradient function for all ReLu activations
    according to Guided Backpropagation.
    """
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)

    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = build_model()
    return new_model


def guided_backprop(input_model, images, layer_name):
    """Guided Backpropagation method for visualizing input saliency."""
    input_imgs = input_model.input
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(layer_output, input_imgs)[0]
    backprop_fn = K.function([input_imgs, K.learning_phase()], [grads])
    grads_val = backprop_fn([images, 0])[0]
    return grads_val


def grad_cam(input_model, image, cls, layer_name):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls]
    #print('input model output:', input_model.output)
    #tf.print('jkf', y_c)
    #exit(1)
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
    # Normalize if necessary
    #grads = normalize(grads)
    gradient_function = K.function([input_model.input], [conv_output, grads])
    
    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]
    
    #print('output', output, 'grads_val', grads_val)
    
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam_max = cam.max() 
    if cam_max != 0: 
        cam = cam / cam_max
    return cam
    
def grad_cam_batch(input_model, images, classes, layer_name):
    """GradCAM method for visualizing input saliency.
    Same as grad_cam but processes multiple images in one run."""
    loss = tf.gather_nd(input_model.output, np.dstack([range(images.shape[0]), classes])[0])
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(loss, layer_output)[0]
    gradient_fn = K.function([input_model.input, K.learning_phase()], [layer_output, grads])

    conv_output, grads_val = gradient_fn([images, 0])    
    weights = np.mean(grads_val, axis=(1, 2))
    cams = np.einsum('ijkl,il->ijk', conv_output, weights)
    
    # Process CAMs
    new_cams = np.empty((images.shape[0], W, H))
    for i in range(new_cams.shape[0]):
        cam_i = cams[i] - cams[i].mean()
        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i, 2) + 1e-10)
        new_cams[i] = cv2.resize(cam_i, (H, W), cv2.INTER_LINEAR)
        new_cams[i] = np.maximum(new_cams[i], 0)
        new_cams[i] = new_cams[i] / new_cams[i].max()
    
    return new_cams

def remove_images(filepath):
    files = glob.glob(filepath + '*.png')
    for f in files:
        #print(f)
        os.remove(f)

def compute_saliency(model, guided_model, img_path, predictions, y, layer_name='block5_conv3', cls=-1, visualize=True, save=True, filename='', num=0):
    """Compute saliency using all three approaches.
        -layer_name: layer to compute gradients;
        -cls: class number to localize (-1 for most probable class).
    """
    preprocessed_input = load_image(img_path)
    #predictions = model.predict(preprocessed_input)
    '''
    top_n = 5
    top = decode_predictions(predictions, top=top_n)[0]
    classes = np.argsort(predictions[0])[-top_n:][::-1]
    print('Model prediction:')
    for c, p in zip(classes, top):
        print('\t{:15s}\t({})\twith probability {:.3f}'.format(p[1], c, p[2]))
    if cls == -1:
        cls = np.argmax(predictions)
    class_name = decode_predictions(np.eye(1, 1000, cls))[0][0][1]
    print("Explanation for '{}'".format(class_name))
    '''
    if cls == -1:
        cls = np.argmax(predictions)
    gradcam = grad_cam(model, preprocessed_input, cls, layer_name)
    #print('\n\n', 'gradcam', gradcam.shape, type(gradcam), np.max(gradcam), np.min(gradcam), '\n\n')
    gb = guided_backprop(guided_model, preprocessed_input, layer_name)
    guided_gradcam = gb * gradcam[..., np.newaxis]
    correctly_classified = False
    #print('y', y)
    #print('pred', predictions)
    if np.argmax(predictions) == np.argmax(y):
        correctly_classified = True
    
    if save:
        jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
        overlay = (np.float32(jetcam) + load_image(img_path, preprocess=False)*255) / 2
        Path(filename + '/gradcam/' + str(args["combined"])).mkdir(parents=True, exist_ok=True)
        Path(filename + '/guided_backprop/' + str(args["combined"])).mkdir(parents=True, exist_ok=True)
        Path(filename + '/guided_gradcam/' + str(args["combined"])).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(os.path.join(filename + '/gradcam/' + str(args["combined"]), str(num) + '_' + str(correctly_classified) + '.png'), np.uint8(overlay))
        cv2.imwrite(os.path.join(filename + '/guided_backprop/' + str(args["combined"]), str(num) + '_' + str(correctly_classified) + '.png'), deprocess_image(gb[0]))
        cv2.imwrite(os.path.join(filename + '/guided_gradcam/' + str(args["combined"]), str(num) + '_' + str(correctly_classified) +'.png'), deprocess_image(guided_gradcam[0]))
    
    if visualize:
        plt.figure(figsize=(15, 10))
        plt.subplot(131)
        plt.title('GradCAM')
        plt.axis('off')
        plt.imshow(load_image(img_path, preprocess=False))
        plt.imshow(gradcam, cmap='jet', alpha=0.5)

        plt.subplot(132)
        plt.title('Guided Backprop')
        plt.axis('off')
        plt.imshow(np.flip(deprocess_image(gb[0]), -1))
        
        plt.subplot(133)
        plt.title('Guided GradCAM')
        plt.axis('off')
        plt.imshow(np.flip(deprocess_image(guided_gradcam[0]), -1))
        plt.show()
        
    return np.uint8(overlay), jetcam, correctly_classified

def make_array(y):
    a = [[0]*6 for i in range(y.shape[0])]
    for i in range(y.shape[0]):
        a[i][y[i][0]] = 1
    return np.asarray(a)

def test(filename):
    model = build_model()
    #(x_train, y_train), (x_test, y_test) = cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = collect_data.Intel.load_data(collect_data.Intel(), train_perc = 80)
    x_train = x_train/255
    x_test = x_test/255
    y_train = make_array(y_train)
    y_test = make_array(y_test)
    '''
    if args["combined"]:
        autoencoder = load_model("saved_models/" + args["runnum"] + "/combined/f_auto.h5")
        x_train = autoencoder.predict(x_train)
        x_test = autoencoder.predict(x_test)
    '''
    guided_model = build_guided_model()
    i=0
    num_samples = 200
    x=x_test[:num_samples]
    y=y_test[:num_samples]
    old_class_predictions = model.predict(x)
    #x = x*255
    Path(filename + '/original/').mkdir(parents=True, exist_ok=True)
    Path(filename + '/collage/').mkdir(parents=True, exist_ok=True)
    '''
    remove_images(filename + '/gradcam/' + str(args["combined"]))
    remove_images(filename + '/guided_backprop/' + str(args["combined"]))
    remove_images(filename + '/guided_gradcam/' + str(args["combined"]))
    remove_images(filename + '/original/')
    remove_images(filename + '/collage/')
    old_class_overlay = []
    old_class_attention = []
    old_class_correct = []
    print('Running old classifier')
    with alive_bar(x.shape[0]) as bar:
        for img in x:
            cv2.imwrite(filename + '/original/' + str(i) + '.png', img*255)
            heatmap, attention, correctly_classified = compute_saliency(model, guided_model, layer_name='block5_conv3',
                                                img_path=img, predictions=predictions[i], y=y[i], cls=-1, visualize=False, save=True, filename=filename,num=i)
            old_class_overlay.append(heatmap)
            old_class_attention.append(attention)
            old_class_correct.append(correctly_classified)
            i=i+1
            bar()
    
    i=0
    print('Running combined')
    '''
    autoencoder = load_model("saved_models/" + args["runnum"] + "/combined/f_auto.h5")
    x_pred = autoencoder.predict(x)
    combined_predictions = model.predict(x_pred)
    '''
    combined_overlay = []
    combined_attention = []
    combined_correct = []
    with alive_bar(x.shape[0]) as bar:
        for img in x_pred:
            #cv2.imwrite(filename + '/original/' + str(i) + '.png', img*255)
            heatmap, attention, correctly_classified = compute_saliency(model, guided_model, layer_name='block5_conv3',
                                                img_path=img, predictions=predictions[i], y=y[i], cls=-1, visualize=False, save=True, filename=filename,num=i)
            combined_overlay.append(heatmap)
            combined_attention.append(attention)
            combined_correct.append(correctly_classified)
            i=i+1
            bar()
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    testlog = open('grad_test_log.txt', 'a')
    testlog.write('\n'+('~'*50)+'\n')
    with alive_bar(x.shape[0]) as bar:
        for i in range(x.shape[0]):
            old_class_overlay, old_class_attention, old_class_correct = compute_saliency(model, guided_model, layer_name='block5_conv3', img_path=x[i], predictions=old_class_predictions[i], y=y[i], cls=-1, visualize=False, save=True, filename=filename,num=i)
            combined_overlay, combined_attention, combined_correct = compute_saliency(model, guided_model, layer_name='block5_conv3', img_path=x_pred[i], predictions=combined_predictions[i], y=y[i], cls=-1, visualize=False, save=True, filename=filename,num=i)
            img1 = old_class_overlay
            img2 = combined_overlay
            img1 = cv2.putText(img1, str(old_class_correct),(5,85), font, 0.4, (255,255,255), 1)
            img2 = cv2.putText(img2, str(combined_correct),(5,85), font, 0.4, (255,255,255), 1)
            row1 = np.hstack([x[i]*255, img1])
            row2 = np.hstack([x_pred[i]*255, img2])
            collage = np.vstack([row1, row2])
            testlog.write('{}\t{}\t{}\n'.format(i, old_class_correct, combined_correct))
            testlog.flush()
            if(old_class_correct != combined_correct):
                #print(i, old_class_correct[i], combined_correct[i])
                cv2.imwrite('{}/collage/diff_{}.png'.format(filename, i), collage)
            else:
                cv2.imwrite('{}/collage/same_{}.png'.format(filename, i), collage)
            bar()
    testlog.close()

if __name__ == '__main__':
    test('Images/gradcam/' + args['runnum'])
    #model = build_model()
    #guided_model = build_guided_model()
    #gradcam, gb, guided_gradcam = compute_saliency(model, guided_model, layer_name='block5_conv3',
                                             #img_path=args["image"], cls=-1, visualize=True, save=True)
