import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow_datasets as tfds
from sklearn.neighbors import KNeighborsClassifier


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
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    score = model.score(X[:2000], y[:2000])
    print(score)    # 14.35%
    score = model.score(X, y)
    print(score)    # 13,5%
    pass


if __name__ == '__main__':
    main()
