# coding=utf-8
import os
import time

import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata
from sklearn.metrics import accuracy_score

mnist_path = os.path.join(os.getcwd(), "mnist")

turn_binary = False
range01_normalize = False
mean_normalize = True
do_PCA = True
n_comp = 50


def get_data():
    '''
    X_train: training image (60000)
    y_train: training label
    X_test: testing image (10000)
    y_test: testing label
    '''
    npz_path = os.path.join(mnist_path, "mnist_data.npz")
    if os.path.isfile(npz_path):
        print("Data already exist!")
        with np.load(npz_path) as data:
            return (data["X_train"], data["y_train"],
                    data["X_test"], data["y_test"])

    print("Start fetching data!")
    mnist = fetch_mldata("MNIST original")
    X, y = mnist.data, mnist.target
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    np.savez(npz_path, X_train=X_train,
             X_test=X_test, y_train=y_train, y_test=y_test)

    return X_train, y_train, X_test, y_test


def main():

    X_train, y_train, X_test, y_test = get_data()

    print("X_train size:", X_train.shape)
    print("y_train size:", y_train.shape)
    print("X_test size:", X_test.shape)
    print("y_test size:", y_test.shape)

    X_train = X_train.astype("float64")
    X_test = X_test.astype("float64")

    # convert pixels to only black(1) or white(0)
    if turn_binary:
        X_train[X_train > 0] = 1
        X_test[X_test > 0] = 1

    # [0, 255] map to [0,1]
    if range01_normalize:
        X_train /= 255.
        X_test /= 255.

    # mean normalization
    if mean_normalize:

        X_train -= np.mean(X_train)
        X_test -= np.mean(X_test)

    # PCA model
    if do_PCA:
        pca_model = PCA(n_components=n_comp, whiten=True, copy=False)
        X_train = pca_model.fit_transform(X_train)
        X_test = pca_model.transform(X_test)

    # build model
    # print("start training!")
    # svm_clf = svm.SVC(C=7, gamma=0.009)
    svm_clf = svm.SVC()
    t0 = time.time()
    svm_clf.fit(X_train, y_train)
    t1 = time.time()
    print("training time: {0:2f} sec".format(t1-t0))
    print("training set acc:", svm_clf.score(X_train, y_train))

    # predict
    # print("start predicting!")
    pred = svm_clf.predict(X_test)
    correct = accuracy_score(pred, y_test, normalize=False)

    print("testing set acc: {0}/{1}({2})".format(
          correct, pred.size, correct/pred.size))


if __name__ == "__main__":
    main()
