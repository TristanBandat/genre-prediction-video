import sys
from os.path import join
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
    train_ds = tfds.load('music4_all_onion_dc:2.0.0', data_dir='data/', batch_size=64,
                         as_supervised=True, split='train')
    # test_ds = tfds.load('music4_all_onion_dc:1.0.1', data_dir='data/', batch_size=64,
    #                     as_supervised=True, split='test')
    # valid_ds = tfds.load('music4_all_onion_dc:1.0.1', data_dir='data/', batch_size=64,
    #                      as_supervised=True, split='valid')
    conv_base = ResNet50(weights='imagenet',
                         include_top=False,
                         input_shape=(64, 64, 3))
    print(conv_base.summary())

    pass


if __name__ == '__main__':
    main()
