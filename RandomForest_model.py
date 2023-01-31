import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow_datasets as tfds
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np


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

    # Number of trees in random forest
    n_estimators = [200, 1000, 2000]
    # Maximum number of levels in tree
    max_depth = [10, 30, 60, None]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    model = RandomForestClassifier()
    model_rand = RandomizedSearchCV(estimator=model, param_distributions=random_grid,
                                    cv=3, n_iter=10, verbose=2, n_jobs=4)
    # Fit the random search model
    model_rand.fit(x_train, y_train)
    print(f'Best Model Parameter: {model_rand.best_params_}')

    model = RandomForestClassifier(model_rand.best_params_)
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
