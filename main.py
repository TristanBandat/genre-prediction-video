import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import datetime
import os


def main():
    # tf.debugging.set_log_device_placement(True)

    train_ds = tfds.load('music4_all_onion_dc:2.1.0', data_dir='data/', batch_size=256,
                         as_supervised=True, split='train')
    test_ds = tfds.load('music4_all_onion_dc:2.1.0', data_dir='data/', batch_size=256,
                        as_supervised=True, split='test')
    valid_ds = tfds.load('music4_all_onion_dc:2.1.0', data_dir='data/', batch_size=256,
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
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=keras.losses.BinaryCrossentropy(),
                  metrics=["accuracy"])
    # keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
    log_dir = os.path.join('logs', 'Simple_NN', 'ResNet', 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    info = model.fit(train_ds, epochs=10, validation_data=valid_ds, callbacks=[tensorboard_callback])
    _, score = model.evaluate(test_ds, callbacks=[tensorboard_callback])
    print(f'Simple_NN (1+1+1): {score}')
    plt.plot(np.arange(1, 11), info.history['loss'], label='Loss', lw=3)
    plt.show()
    plt.plot(np.arange(1, 11), info.history['accuracy'], label='Accuracy', lw=3)
    plt.show()
    pass


if __name__ == '__main__':
    main()
