import pandas as pd
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def create_model():
    # define a sequential model
    model = tf.keras.models.Sequential([
        layers.Embedding(input_dim=4096, output_dim=1024),
        # layers.LSTM(128),
        layers.LSTM(128, return_sequences=True),
        # layers.LSTM(128),
        layers.LSTM(64),
        # layers.Dense(1),
        layers.Dense(1000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(685, activation='sigmoid'),

    ])

    return model


def main():
    train_ds = tfds.load('music4_all_onion_dc:1.0.1', data_dir='data/', batch_size=64,
                         as_supervised=True, split='train')
    # test_ds = tfds.load('music4_all_onion_dc:1.0.1', data_dir='data/', batch_size=64,
    #                     as_supervised=True, split='test')
    # valid_ds = tfds.load('music4_all_onion_dc:1.0.1', data_dir='data/', batch_size=64,
    #                      as_supervised=True, split='valid')
    # input_layer = keras.Input(shape=(4096,), name='input')
    # hidden_layer = layers.Dense(32000, activation='relu')(input_layer)
    # output_layer = layers.Dense(685, activation='sigmoid')(hidden_layer)
    # model = keras.Model(input_layer, output_layer)
    model = create_model()
    model.compile("adam", "mean_squared_error", metrics=["accuracy"])
    # keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
    model.fit(train_ds, epochs=2)
    # around 7% accuracy
    pass


if __name__ == '__main__':
    main()
