# coding=utf-8
import os
import logging
from time import time

import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import utility as ut
import feature_extractor as FE

DATA_PATH = os.path.join(os.getcwd(), "FaceRecognizer", "data")
# DATA_PATH = os.path.join(os.getcwd(), "data")
FACE_PATH = os.path.join(DATA_PATH, "face_database")
LABELED_PATH = os.path.join(DATA_PATH, "labeled_face.npz")
MODEL_PATH = [os.path.join(DATA_PATH, "pca_model.pkl"), os.path.join(DATA_PATH, "pca_svm_model.pkl"), os.path.join(
    DATA_PATH, "hog_svm_model.pkl")]

SPLIT_RATIO = 0.1
VALIDATION = False

PCA_N = 160
# INFO:root:mean_width: 142
MEAN_WIDTH = 142
# INFO:root:mean_height: 208
MEAN_HEIGHT = 208


def get_labeled_data(flip=False, gray=False):
    '''
    Generate labeled data from raw image with filename: sxx_yy.jpg.
    xx for labeled.

    Param:
        flip: 
            True, will add image flipped horizontally to dataset.
            False, no flipped image.
        gray:
            True, read the grayscale image.
            False, read the color(BGR) image.
    Return:
        (images, labels)
        ndarray of images(BGR) and labels(1D)
    '''
    # load raw data
    filenames = os.listdir(FACE_PATH)
    count = len(filenames)
    mean_width, mean_height = 0, 0
    imgs, labels = [], []

    for s in filenames:
        # convert to gray scale
        img = cv.imread(os.path.join(FACE_PATH, s),
                        cv.IMREAD_GRAYSCALE if gray else cv.IMREAD_COLOR)
        label = int(s.split("_")[0][1:])
        mean_width += img.shape[1]
        mean_height += img.shape[0]
        imgs.append(img)
        labels.append(label)

        # flip horizontally
        if flip:
            img = cv.flip(img, 1)
            imgs.append(img)
            labels.append(label)
            # ut.show_img(img, to_RGB=False,gray=True)
        # logging.info(s+" : " + str(label))

    mean_width //= count
    mean_height //= count

    logging.info("mean_width: "+str(mean_width))
    logging.info("mean_height: "+str(mean_height))

    for i in range((count*2) if flip else count):
        imgs[i] = cv.resize(imgs[i], (mean_width, mean_height),
                            interpolation=cv.INTER_LINEAR)
        # ut.show_img(img, to_RGB=True,gray=False)

    return np.array(imgs, dtype=np.uint8), np.array(labels, dtype=np.uint8)


def train_pca_svm_model():
    '''
    Use grayscale images and applying histogram equalization to them.
    Then, PCA for dimensionality reduction and SVM for training.

    Param:
        X: ndarray of images.
        Y: ndarray of labels.

    Return:
        (pca model, svm model)        
    '''
    print("="*10 + " Start construct PCA+SVM model " + "="*10)
    # Generate labeled data
    print("Generate labeled data...")
    t0 = time()
    imgs, labels = get_labeled_data(flip=True, gray=True)
    print("total image: "+str(imgs.shape[0]))
    print("total label: "+str(labels.shape[0]))
    # print("image shape: {0} x {1} x {2} (W x H x COLOR)".format(imgs.shape[2], imgs.shape[1], imgs.shape[3]))
    print("done in {0:.3f}s".format(time() - t0))

    # Split into a training and testing set
    if VALIDATION:
        X_train, X_test, Y_train, Y_test = train_test_split(
            imgs, labels, test_size=SPLIT_RATIO, shuffle=True)
    else:
        X_train, Y_train = imgs, labels

    # Conver 2D image to 1D array
    X_train = X_train.reshape((X_train.shape[0], -1))

    # PCA for dimensionality reduction
    pca = FE.get_PCA_model(X_train, n_comp=PCA_N)

    # Use PCA on training data
    print("Use PCA on data...")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    print("done in {0:.3f}s".format(time() - t0))
    print("shape of PCA feature vectors: "+str(X_train_pca.shape))

    # Train a SVM classification model
    print("Fitting the classifier(PCA+SVM) to the training set..")
    t0 = time()
    param_grid = {'C': [1, 1e3, 5e3, 1e4],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005]}
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_pca, Y_train)
    print("done in {0:.3f}s".format(time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    print("Training set acc:", clf.score(X_train_pca, Y_train))

    # Preprocess test set
    if VALIDATION:
        print("Testing set:")
        X_test = X_test.reshape((X_test.shape[0], -1))
        X_test_pca = pca.transform(X_test)
        print("shape of feature vectors: "+str(X_test_pca.shape))
        # Predict on the testing set
        print("Predicting on the testing set...")
        t0 = time()
        Y_pred = clf.predict(X_test_pca)
        print("done in {0:.3f}s".format(time() - t0))
        print("testing set acc: {0}".format(
            accuracy_score(Y_test, Y_pred, normalize=True)))

    # print(classification_report(Y_test, Y_pred))
    # np.savez(LABELED_PATH, X=X, Y=Y)
    print("="*15 + " DONE! " + "="*15)
    return pca, clf


def train_hog_svm_model():

    print("="*10 + " Start construct HOG+SVM model " + "="*10)

    # Generate labeled data
    print("Generate labeled data...")
    t0 = time()
    imgs, labels = get_labeled_data(flip=True, gray=False)
    print("total image: "+str(imgs.shape[0]))
    print("total label: "+str(labels.shape[0]))
    print("image shape: {0} x {1} x {2} (W x H x COLOR)".format(
        imgs.shape[2], imgs.shape[1], imgs.shape[3]))
    print("done in {0:.3f}s".format(time() - t0))

    # Split into a training and testing set
    if VALIDATION:
        X_train, X_test, Y_train, Y_test = train_test_split(
            imgs, labels, test_size=SPLIT_RATIO, shuffle=True)
    else:
        X_train, Y_train = imgs, labels

    X_train_hog = FE.get_HOG_vec(X_train)

    # Train a SVM classification model
    print("Fitting the classifier(HOG+SVM) to the training set..")
    t0 = time()
    param_grid = {'C': [1, 1e3, 5e3, 1e4]}
    clf = GridSearchCV(LinearSVC(class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_hog, Y_train)
    print("done in {0:.3f}s".format(time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    print("Training set acc:", clf.score(X_train_hog, Y_train))

    if VALIDATION:
        print("Testing set:")
        X_test_hog = FE.get_HOG_vec(X_test)
        # Predict on the testing set
        print("Predicting on the testing set...")
        t0 = time()
        Y_pred = clf.predict(X_test_hog)
        print("done in {0:.3f}s".format(time() - t0))
        print("testing set acc: {0}".format(
            accuracy_score(Y_test, Y_pred, normalize=True)))

    print("="*15 + " DONE! " + "="*15)
    return clf


def main():
    train_pca_svm_model()
    train_hog_svm_model()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
