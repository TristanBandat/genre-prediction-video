import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow_datasets as tfds
from sklearn.neighbors import KNeighborsClassifier


def main():
    train_ds = tfds.load('music4_all_onion_dc:1.0.1', data_dir='data/', batch_size=1,
                         as_supervised=True, split='train')
    test_ds = tfds.load('music4_all_onion_dc:1.0.1', data_dir='data/', batch_size=1,
                        as_supervised=True, split='test')
    valid_ds = tfds.load('music4_all_onion_dc:1.0.1', data_dir='data/', batch_size=1,
                         as_supervised=True, split='valid')

    xy_train = list(train_ds.as_numpy_iterator())
    x_train = [x[0][0] for x in xy_train]
    y_train = [x[1][0] for x in xy_train]
    xy_test = list(test_ds.as_numpy_iterator())
    x_test = [x[0][0] for x in xy_test]
    y_test = [x[1][0] for x in xy_test]
    xy_valid = list(valid_ds.as_numpy_iterator())
    x_valid = [x[0][0] for x in xy_valid]
    y_valid = [x[1][0] for x in xy_valid]

    num_neighbors_list = range(2, 10)
    train_scores = list()
    test_scores = list()
    model_list = list()

    for num_neighbors in num_neighbors_list:
        model = KNeighborsClassifier(n_neighbors=num_neighbors)
        model.fit(x_train, y_train)
        train_scores.append(model.score(x_train[:2000], y_train[:2000]))
        print(f'Train score ({num_neighbors}): {train_scores[-1]}')
        test_scores.append(model.score(x_test, y_test))
        print(f'Test score ({num_neighbors}): {test_scores[-1]}')
        model_list.append(model)
    for i, model in enumerate(model_list):
        print(f'Model {i+1}: {model.score(x_valid, y_valid)}')
    pass


if __name__ == '__main__':
    main()
