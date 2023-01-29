import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras.utils import plot_model
import datetime
import os


def create_model():
    model = keras.Sequential()
    model.add(layers.Dense(4096, input_shape=(4096,), activation='relu'))
    for i in range(10):
        model.add(layers.Dense(6500, activation='relu'))
    model.add(layers.Dense(53, activation='sigmoid'))

    return model


def main():
    # tf.debugging.set_log_device_placement(True)
    train_ds = tfds.load('music4_all_onion_dc:2.1.0', data_dir='data/', batch_size=64,
                         as_supervised=True, split='train')
    test_ds = tfds.load('music4_all_onion_dc:2.1.0', data_dir='data/', batch_size=64,
                        as_supervised=True, split='test')
    valid_ds = tfds.load('music4_all_onion_dc:2.1.0', data_dir='data/', batch_size=64,
                         as_supervised=True, split='valid')

    model = create_model()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss=keras.losses.BinaryCrossentropy(),
                  metrics=["accuracy"])
    # keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
    log_dir = os.path.join('logs', 'Deep_NN', 'ResNet', 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(train_ds, epochs=10, validation_data=valid_ds, callbacks=[tensorboard_callback])
    _, score = model.evaluate(test_ds, callbacks=[tensorboard_callback])
    print(f'Simple_NN (1+10+1): {score}')
    pass


if __name__ == '__main__':
    main()
