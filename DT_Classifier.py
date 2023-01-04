import sklearn
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
# from tensorflow.keras import layers
from sklearn.tree import DecisionTreeClassifier


def main():
    # train_ds = tfds.load('music4_all_onion_dc:1.0.1', data_dir='data/', batch_size=1,
    #                      as_supervised=True, split='train')
    test_ds = tfds.load('music4_all_onion_dc:1.0.1', data_dir='data/', batch_size=1,
                        as_supervised=True, split='test')
    # valid_ds = tfds.load('music4_all_onion_dc:1.0.1', data_dir='data/', batch_size=1,
    #                      as_supervised=True, split='valid')

    Xy = list(test_ds.as_numpy_iterator())
    X = [x[0][0] for x in Xy]
    y = [x[1][0] for x in Xy]
    model = DecisionTreeClassifier(random_state=0, max_depth=10)
    model.fit(X[:1000], y[:1000])
    score = model.score(X[1000:2000], y[1000:2000])
    print(score)
    pass


if __name__ == '__main__':
    main()
