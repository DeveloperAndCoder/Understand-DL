import numpy as np
from keras.datasets import cifar10
from keras.models import load_model
import matplotlib.pyplot as plt
import os
import cv2
import sys
from STL10 import stl_to_dataset

if(len(sys.argv) != 2) :
    print('There need to be only one argument - Run number given')
    exit(1)

runnum = str(sys.argv[1])
runnum.strip()
print("runnum:", runnum)

save_dir = "saved_models/{}/".format(runnum)

def make_array(y):
    a = [[0]*10 for i in range(y.shape[0])]
    for i in range(y.shape[0]):
        a[i][y[i]] = 1
    return np.asarray(a)


autoencoder = load_model(save_dir+'/autoencoder.h5')
classifier = load_model(save_dir+'/classifier.h5')
f_auto = load_model(save_dir+'/combined/f_auto.h5')
f_class = load_model(save_dir+'/combined/f_class.h5')
f_combined = load_model(save_dir+'/combined/f_combined.h5')

print('Loaded all models')

(x_train, y_train), (x_test, y_test) = stl_to_dataset.get_numpy(ratio=0.8) #ratio in which to split train and test
print('Number of test samples:', y_test.shape[0])
x_train = x_train/255
x_test = x_test/255
y_train = make_array(y_train)
y_test = make_array(y_test)

#num_samples = 10
sample = x_test#[:num_samples]
img_init = autoencoder.predict(sample) #output of pretrained autoencoder
img_final = f_auto.predict(sample)     #output of finetuned autoencoder

'''
diff = np.absolute(img_init-img_final)*255
for i in range(num_samples):
    cv2.imwrite('res{:03d}.png'.format(i), diff[i])
#print(pred.shape, pred[0])
#cv2.imwrite('res0.png', pred[0]*255)
'''

pred_init = classifier.predict(x_test)  #classifier on original images
pred_man = classifier.predict(img_init) #classifier on images from pretrained autoencoder
pred_comb = f_combined.predict(x_test)  #classifier on images from finetuned autoencoder
#pred_f2 = f_class.predict(x_test)

mse_init = (np.square(pred_init-y_test)).mean(axis=None)
mse_man = (np.square(pred_man-y_test)).mean(axis=None)
mse_comb = (np.square(pred_comb-y_test)).mean(axis=None)


#mse_f2 = (np.square(pred_f2-y_test)).mean(axis=None)
print('Pretrained classifer on original images\t\t\t\t{}\nPretrained classifier on images from pretrained autoencoder\t{}\nPretrained classifier on images from finetuned autoencoder\t{}'
       .format(mse_init, mse_man, mse_comb))
