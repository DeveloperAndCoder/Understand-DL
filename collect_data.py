import os
import cv2
import numpy as np
import json
from PIL import Image
import keras.utils as keras_utils

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

def make_array(y, num_of_classes=10):
    a = [[0]*num_of_classes for i in range(y.shape[0])]
    for i in range(0,y.shape[0]):
        a[i][y[i][0]] = 1
    return np.asarray(a)

def pre_process(x_train, y_train, x_test, y_test):
    y_train = make_array(y_train)
    y_test = make_array(y_test)
    #y_train = keras.utils.to_categorical(y_train, num_of_classes)
    #y_test = keras.utils.to_categorical(y_test, num_of_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return (x_train, y_train), (x_test, y_test)

=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
class Imagenet:
    image_dir = 'imagenet/images/'

    def set_image_dir(self, dir_path):
        self.image_dir = dir_path

    def resize(self, img, dims = (32, 32)):
        return cv2.resize(img, dims)

    def make_map_from_wnid_to_classno(self):
        map_from_wnid_to_classno = {}
        CLASS_INDEX_PATH = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
        fpath = keras_utils.get_file(
            'imagenet_class_index.json',
            CLASS_INDEX_PATH)
        CLASS_INDEX = None
        with open(fpath) as f:
            CLASS_INDEX = json.load(f)
        for i in range(0,1000) :
            # print('mapping', CLASS_INDEX[str(i)][0], i)
            map_from_wnid_to_classno[CLASS_INDEX[str(i)][0]] = i
        return map_from_wnid_to_classno

    def load_data(self, train_perc = 80, toResize = False, dims = (32, 32)):
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        map_from_wnid_to_classno = self.make_map_from_wnid_to_classno()
        for wnid in os.listdir(self.image_dir):
            print("Reading images of "  + wnid)
            #print(len([img for img in os.listdir(self.image_dir + category)]))
            tot = len([img for img in os.listdir(self.image_dir + wnid)])
            print("Total = " + str(tot))
            taken = 0
            for img in os.listdir(self.image_dir + wnid):
                #if (img[-3:] == 'jpg' or img[-3:] == 'png' or img[-4:] == 'jpeg'):
                #print(img, taken)
                full_addr = self.image_dir + wnid + "/" + img
                im = cv2.imread(full_addr)
                if(type(im) != np.ndarray):
                    #print(type(im)) NoneType
                    continue
                #print(full_addr + " type = ", type(im))
                if (toResize) :
                    im = self.resize(im, dims)
                #print(im.shape)
                if (taken * 100 <= tot * train_perc):
                    x_train.append(im)
                    y_train.append(map_from_wnid_to_classno[wnid])
                    taken += 1
                else :
                    x_test.append(im)
                    y_test.append(map_from_wnid_to_classno[wnid])
                    #print(y_test[-1])
            print("Done reading category ", wnid)
            break
        print("(length of x_train), (length of x_test)", len(x_train), len(x_test))
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_train = y_train.reshape((y_train.shape[0]), 1)
        y_test = y_test.reshape((y_test.shape[0]), 1)
        print('x_train', type(x_train), x_train.shape, x_train[0].shape)
        return (x_train, y_train), (x_test, y_test)

class STL10:
    image_dir = 'STL10/img/'
    def set_image_dir(self, dir_path):
        self.image_dir = dir_path

    def resize(self, img, dims = (32, 32)):
        return img.resize(dims)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    def load_data(self, train_perc = 80, toResize=False, dims=(32,32), preprocess = False):
=======
    def load_data(self, train_perc = 80, toResize=False, dims=(32,32)):
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
    def load_data(self, train_perc = 80, toResize=False, dims=(32,32)):
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
    def load_data(self, train_perc = 80, toResize=False, dims=(32,32)):
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
    def load_data(self, train_perc = 80, toResize=False, dims=(32,32)):
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
        classes = os.listdir(self.image_dir)

        x_train = []
        y_train = []
        x_test = []
        y_test = []

        for c in classes:
            taken = 0
            imgs = os.listdir(self.image_dir + c)
            tot = len([img for img in os.listdir(self.image_dir + c)])
            for img in imgs:
                im = Image.open(self.image_dir + c + '/' + img)
                if (toResize) :
                    im = self.resize(im, dims)
                arr = np.asarray(im)
                #print(im.shape)
                if (taken * 100 <= tot * train_perc):
                    x_train.append(arr)
                    y_train.append(int(c)-1)
                    taken += 1
                else :
                    x_test.append(arr)
                    y_test.append(int(c)-1)
                    # print(y_test[-1])

        print("Done loading")
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_train = y_train.reshape((y_train.shape[0]), 1)
        y_test = y_test.reshape((y_test.shape[0]), 1)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        print('x_train kundli', type(x_train), x_train.shape, x_train[0].shape)

        if preprocess:
            # pre_process(x_train, y_train, x_test, y_test)
            (x_train, y_train), (x_test, y_test) = pre_process(x_train, y_train, x_test, y_test)
=======
        # print(type(x_train), x_train.shape, x_train[0].shape)
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
        # print(type(x_train), x_train.shape, x_train[0].shape)
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
        # print(type(x_train), x_train.shape, x_train[0].shape)
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
        # print(type(x_train), x_train.shape, x_train[0].shape)
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377

        return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    # Imagenet.make_map_from_wnid_to_classno(Imagenet())
    (x_train, y_train), (x_test, y_test) = STL10.load_data(STL10(), toResize=False)
    # im = cv2.imread('Images/1.jpg')
    # im = Imagenet.resize(Imagenet(), im)
    # print(im.shape)
    # cv2.imshow('cat', im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
