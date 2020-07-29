from unet_model import *
import collect_data
import argparse
from pathlib import Path
import os
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.datasets import cifar10
from keras.utils import plot_model
import cv2

#os.environ["CUDA_VISIBLE_DEVICES"]="7"

model = unet(input_size=(96,96,3))
# exit(1)
plot_model(model, show_shapes=True, to_file='unet_arch.png')

# testGene = testGenerator("data/membrane/test")
# results = model.predict_generator(testGene,30,verbose=1)
# saveResult("data/membrane/test",results)
