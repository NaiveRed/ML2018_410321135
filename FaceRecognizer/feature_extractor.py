# coding=utf-8
import logging
from time import time

import numpy as np
from sklearn.decomposition import PCA
from skimage.feature import hog

import utility

# PCA
PCA_N = 160

# HOG parameter for feature extractor
ORIENT = 9  # number of bins for HOG
PIXELS_PER_CELL = (32, 32)  # 32x32
CELLS_PER_BLOCK = (2, 2)  # 2x2 cell
BLOCK_NORM = "L2"


def get_PCA_model(X, n_comp=PCA_N):

    print("Compute PCA...")
    t0 = time()
    pca = PCA(n_components=PCA_N, svd_solver='randomized',
              whiten=True).fit(X)
    print("done in {0:.3f}s".format(time() - t0))

    return pca


def get_HOG_vec(imgs):
    ''''
    Extract HOG features vector from images

    Param
        data: ndarray of images

    return: ndarray of HOG features vectors
    '''

    # HOG_test(imgs[0])
    features = []

    # process data
    for img in imgs:
        fv = hog(img, orientations=ORIENT, pixels_per_cell=PIXELS_PER_CELL,
                 cells_per_block=CELLS_PER_BLOCK, block_norm=BLOCK_NORM,
                 feature_vector=True, visualize=False, multichannel=True)

        features.append(fv)

    features = np.array(features)

    # print("shape of HOG feature vectors: "+str(features.shape))

    return features


def HOG_test(img):
    '''
    Show the HOG of one image.
    '''
    # HOG descriptor, visualisation of the HOG image
    fd, hog_image = hog(img, orientations=ORIENT, pixels_per_cell=PIXELS_PER_CELL,
                        cells_per_block=CELLS_PER_BLOCK, block_norm=BLOCK_NORM,
                        feature_vector=True, visualize=True, multichannel=True)

    # show result
    utility.show_subplot(1, 2, "Result", [img, hog_image], [
                         "Input", "Histogram of Oriented Gradients"])
