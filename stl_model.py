import keras
from keras.layers import Dense, Flatten, Input, Dropout
#import tensorflow.keras.preprocessing.image.image_dataset_from_directory
from keras.applications import VGG16
from keras.models import Sequential, Model
import config

baseModel = VGG16(
    weights="imagenet",
    include_top=False,
    # pooling='max',
    input_shape = (96,96,3)
    )

baseModel.summary()

headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(config.num_of_classes, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)