# THIS CODE IS A MODULARIZED AND UPDATED VERSION OF CODE FROM
# "Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow" by Aurélien Geron
# https://github.com/ageron/handson-ml2/blob/master/08_dimensionality_reduction.ipynb
# specifically Exercise 9 of this notebook

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from ptmlib.time import Stopwatch
from ptmlib.cpu import CpuCount


def get_data():
    from sklearn.datasets import fetch_openml

    train_data_count = 60000
    print('\n', 'RETRIEVING MNIST_784 DATA', '\n')
    mnist = fetch_openml('mnist_784', version=1)
    mnist.target = mnist.target.astype(np.uint8)

    x_train = mnist['data'][:train_data_count]
    y_train = mnist['target'][:train_data_count]

    x_test = mnist['data'][train_data_count:]
    y_test = mnist['target'][train_data_count:]

    return (x_train, y_train), (x_test, y_test)


def get_reduced_data(x_train, x_test, variance):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=variance)
    x_train_reduced = pca.fit_transform(x_train)

    print('x_train shape:', x_train.shape)
    print('x_train_reduced shape:', x_train_reduced.shape)

    x_test_reduced = pca.transform(x_test)

    return x_train_reduced, x_test_reduced


def main():

    # INFO AND SETUP
    print('Scikit-Learn Version:', sklearn.__version__)
    stopwatch = Stopwatch()
    cpu_count = CpuCount()
    cpu_count.print_stats()
    max_cpu = cpu_count.adjusted_count_by_percent()  # don't use too many processors :)

    # HYPER PARAMS
    hp_estimators = 100
    hp_random_state = 42  # i really should read this book already
    hp_variance = 0.95

    (x_train, y_train), (x_test, y_test) = get_data()

    # TRAINING
    # we can speed up RandomForestClassifier by using multiple CPU cores for concurrent processing
    print(f'Init RandomForestClassifier with n_jobs={max_cpu}')
    rnd_clf = RandomForestClassifier(n_estimators=hp_estimators, random_state=hp_random_state, n_jobs=max_cpu)

    stopwatch.start()
    rnd_clf.fit(x_train, y_train)
    stopwatch.stop()

    y_pred = rnd_clf.predict(x_test)

    print('\n', 'INITIAL ACCURACY SCORE:', accuracy_score(y_test, y_pred), '\n')

    # USE PCA TO REDUCE DIMENSIONALITY; COMPARE SPEED/ACCURACY
    x_train_reduced, x_test_reduced = get_reduced_data(x_train, x_test, hp_variance)
    rnd_clf = RandomForestClassifier(n_estimators=hp_estimators, random_state=hp_random_state, n_jobs=max_cpu)

    stopwatch.start()
    rnd_clf.fit(x_train_reduced, y_train)
    stopwatch.stop()

    y_pred = rnd_clf.predict(x_test_reduced)

    print('\n', 'PCA ACCURACY SCORE:', accuracy_score(y_test, y_pred), '\n')
    
    print('EXPECTED RESULT: SLOWER training and LOWER accuracy; PCA is not always the answer')
    print('See exercise 9 in the following notebook for more info:')
    print('https://github.com/ageron/handson-ml2/blob/master/08_dimensionality_reduction.ipynb')


if __name__ == '__main__':
    main()
