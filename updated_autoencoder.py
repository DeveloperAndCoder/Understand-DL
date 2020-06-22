import numpy as np
from keras.layers import Conv2D, UpSampling2D, BatchNormalization
from keras.models import Sequential
from keras.utils import multi_gpu_model
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import os
import sys
import collect_data
from pathlib import Path


def _get_available_gpus():
    
    global _LOCAL_DEVICES
    if _LOCAL_DEVICES is None:
        if _is_tf_1():
            devices = get_session().list_devices()
            _LOCAL_DEVICES = [x.name for x in devices]
        else:
            devices = tf.config.list_logical_devices()
            _LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in _LOCAL_DEVICES if 'device:gpu' in x.lower()]


if(len(sys.argv) != 2) :
    print('There need to be only one argument - Run number given')
    exit(1)

runnum = str(sys.argv[1])
print("runnum = ", runnum)

Path("saved_models/" + runnum).mkdir(parents=True, exist_ok=True)

save_dir = os.path.join(os.getcwd(), 'saved_models/' +runnum)
model_name = 'autoencoder.h5'

(x_train, _), (x_test, _) = cifar10.load_data()
# (x_train, y_train), (x_test, y_test) = collect_data.Imagenet.load_data(collect_data.Imagenet(), toResize=True)
x_train = x_train/255
x_test = x_test/255


# The next three methods to visualize input/output of our model side-by-side
def hstackimgs(min, max, images):
    return np.hstack(images[i] for i in range(min, max))

def sqstackimgs(length, height, images):
    return np.vstack(hstackimgs(i*length, (i+1)*length, images) for i in range(height))

def sbscompare(images1, images2, length, height):
    A = sqstackimgs(length, height, images1)
    B = sqstackimgs(length, height, images2)
    C = np.ones((A.shape[0], 32, 3))
    return np.hstack((A, C, B))



model = Sequential()

model.add(Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(224, 224, 3)))
model.add(BatchNormalization())     # 32x32x32 # 224x224x16
model.add(Conv2D(16, kernel_size=3, strides=2, padding='same', activation='relu'))      # 16x16x32 112x112x16
model.add(Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu'))      # 16x16x32 112x112x16
model.add(BatchNormalization())     # 32x32x32 # 112x112x16
model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'))      # 56x56x32
model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))      # 56x56x32
model.add(BatchNormalization())     # 32x32x32 # 56x56x32
model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu'))      # 28x28x64
model.add(Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'))      # 28x28x64
model.add(BatchNormalization())     # 23x23x64
model.add(Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu'))      # 14x14x128
model.add(Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'))      # 14x14x128
model.add(BatchNormalization())     # 23x23x64
model.add(Conv2D(256, kernel_size=3, strides=2, padding='valid', activation='relu'))      # 7x7x256
model.add(Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'))      # 7x7x256
model.add(BatchNormalization())     # 7x7x256

model.add(UpSampling2D())
model.add(Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'))      # 32x32x32 14x14x128
model.add(BatchNormalization())
model.add(UpSampling2D())
model.add(Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'))      # 32x32x32 28x28x64
model.add(BatchNormalization())
model.add(UpSampling2D())
model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))      # 32x32x32 56x56x32
model.add(BatchNormalization())
model.add(UpSampling2D())
model.add(Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu'))      # 32x32x32 112x112x16
model.add(BatchNormalization())
model.add(UpSampling2D())
model.add(Conv2D(8, kernel_size=3, strides=1, padding='same', activation='relu'))      # 32x32x32 224x224x8
model.add(BatchNormalization())

model.add(Conv2D(3,  kernel_size=1, strides=1, padding='same', activation='sigmoid'))   # 32x32x3 224x224x3

model.compile(optimizer='adam', metrics=['accuracy'], loss='mean_squared_error')
model.summary()

#parallel_model = multi_gpu_model(model, gpus=4)
#parallel_model.compile(loss='mean_squared_error', optimizer='adam')

# We want to add different noise vectors for each epoch
num_epochs = 50
#NOISE = 0.3     # Set to 0 for a regular (non-denoising...) autoencoder
#for i in range(num_epochs):
    #noise = np.random.normal(0, NOISE, x_train.shape)
    

#parallel_model.fit(x_train, x_train, epochs=num_epochs, batch_size=100)
model.fit(x_train, x_train, epochs=num_epochs, batch_size=100)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
x_test = x_test[:400]
#noise = np.random.normal(0, NOISE, x_test.shape)
pred_imgs = model.predict(x_test)

plt.imshow(sbscompare(x_test, pred_imgs, 20, 20))
plt.axis('off')
plt.rcParams["figure.figsize"] = [60,60]
plt.savefig('result.png', dpi=200)
plt.clf()

