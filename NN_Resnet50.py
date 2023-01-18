import sys
from os.path import join

import keras.callbacks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

from keras import models, regularizers, layers, optimizers, losses, metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications import ResNet50


def main():
    model_path = 'tl_resnet_weights.h5'

    train_ds = tfds.load('music4_all_onion_dc:3.0.2', data_dir='data/', batch_size=32,
                         as_supervised=True, split='train')
    test_ds = tfds.load('music4_all_onion_dc:3.0.2', data_dir='data/', batch_size=32,
                        as_supervised=True, split='test')
    valid_ds = tfds.load('music4_all_onion_dc:3.0.2', data_dir='data/', batch_size=32,
                         as_supervised=True, split='valid')

    conv_base = ResNet50(weights='imagenet',
                         include_top=False,
                         input_shape=(64, 64, 3))
    # print(conv_base.summary())
    for layer in conv_base.layers[:143]:
        layer.trainable = False
    # for i, layer in enumerate(conv_base.layers):
    #     print(f"{i} {layer.name} - {layer.trainable}")

    model = Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(685, activation='sigmoid'))

    # check_point = keras.callbacks.ModelCheckpoint(filepath=model_path,
    #                                               monitor='val_acc',
    #                                               mode='max',
    #                                               save_best_only=True)

    model.compile("adam", loss=keras.losses.BinaryCrossentropy(), metrics=["accuracy"])
    # plot_model(model, show_shapes=True, rankdir="LR")
    model.fit(train_ds, epochs=10, verbose=1, validation_data=test_ds)
    _, score = model.evaluate(valid_ds)
    print(f'ResNet TL NN (vgg19 with 64x64x3): {score}')
    model.summary()
    model.save(model_path)

    pass


if __name__ == '__main__':
    main()
