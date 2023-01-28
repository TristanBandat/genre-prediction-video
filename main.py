import pandas as pd
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def main():
    # tf.debugging.set_log_device_placement(True)

    train_ds = tfds.load('music4_all_onion_dc:1.1.0', data_dir='data/', batch_size=64,
                         as_supervised=True, split='train')
    test_ds = tfds.load('music4_all_onion_dc:1.1.0', data_dir='data/', batch_size=64,
                        as_supervised=True, split='test')
    valid_ds = tfds.load('music4_all_onion_dc:1.1.0', data_dir='data/', batch_size=64,
                         as_supervised=True, split='valid')
    # plt.figure(figsize=(10, 10))
    # for element in ds:
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.plot(element['input'][i].numpy())
    #         plt.title(f'Input: {i}')
    #         plt.axis("off")
    #     break
    # plt.show()

    input_layer = keras.Input(shape=(4096,), name='input')
    hidden_layer = layers.Dense(32000, activation='relu')(input_layer)
    output_layer = layers.Dense(53, activation='sigmoid')(hidden_layer)
    model = keras.Model(input_layer, output_layer)
    model.compile("adam", "mean_squared_error", metrics=["accuracy"])
    # keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
    info = model.fit(train_ds, epochs=10)
    print(info)
    pass


if __name__ == '__main__':
    main()
