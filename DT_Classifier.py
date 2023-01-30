import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow_datasets as tfds
from sklearn.tree import DecisionTreeClassifier


def main():
    train_ds = tfds.load('music4_all_onion_dc:1.1.0', data_dir='data/', batch_size=1,
                         as_supervised=True, split='train')
    test_ds = tfds.load('music4_all_onion_dc:1.1.0', data_dir='data/', batch_size=1,
                        as_supervised=True, split='test')
    valid_ds = tfds.load('music4_all_onion_dc:1.1.0', data_dir='data/', batch_size=1,
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

    model = DecisionTreeClassifier(max_depth=10)
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    print(f'Score on train set: {train_score}')
    test_score = model.score(x_test, y_test)
    print(f'Score on test set: {test_score}')
    valid_score = model.score(x_valid, y_valid)
    print(f'Score on valid set: {valid_score}')
    pass


if __name__ == '__main__':
    main()
