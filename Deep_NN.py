import pandas as pd
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def create_model():
    model = keras.Sequential()
    model.add(layers.Dense(4096, input_shape=(4096,), activation='relu'))
    for i in range(10):
        model.add(layers.Dense(10000, activation='relu'))
    model.add(layers.Dense(685, activation='sigmoid'))

    return model


def main():
    # tf.debugging.set_log_device_placement(True)

    train_ds = tfds.load('music4_all_onion_dc:1.0.1', data_dir='data/', batch_size=64,
                         as_supervised=True, split='train')
    test_ds = tfds.load('music4_all_onion_dc:1.0.1', data_dir='data/', batch_size=64,
                        as_supervised=True, split='test')
    # valid_ds = tfds.load('music4_all_onion_dc:1.0.1', data_dir='data/', batch_size=64,
    #                      as_supervised=True, split='valid')
    model = create_model()
    model.compile("adam", "mean_squared_error", metrics=["accuracy"])
    keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
    model.fit(train_ds, epochs=10)
    pass


if __name__ == '__main__':
    main()
