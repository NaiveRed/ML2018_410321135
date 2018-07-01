# coding=utf-8
import os
import sys
import pickle
from time import time

import cv2 as cv

import training
import utility as ut
import feature_extractor as FE

help_str = """
Usage:

    Train the model:
        python face_recognizer.py train

    Predict the image in data/test_img:
        python face_recognizer.py pred
"""

TEST_PATH = os.path.join(training.DATA_PATH, "test_img")
DEBUG = False


def data_process(img_path):
    '''
    Return:
        (grayscale, grayscale and filp horizontally,
        color(BGR), color(BGR) and filp horizontally)
    '''
    # convert to gray scale
    gray_img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    color_img = cv.imread(img_path, cv.IMREAD_COLOR)

    # flip horizontally
    gray_fimg = cv.flip(gray_img, 1)
    color_fimg = cv.flip(color_img, 1)
    res = [gray_img, gray_fimg, color_img, color_fimg]

    # ut.show_img(img, to_RGB=False,gray=True)
    for i in range(4):
        res[i] = cv.resize(res[i], (training.MEAN_WIDTH,
                                    training.MEAN_HEIGHT), interpolation=cv.INTER_LINEAR)
    # ut.show_img(res[2], to_RGB=True, gray=False)

    return res


def main():
    command = []
    if DEBUG:
        command = ["train"]
    elif len(sys.argv) > 1:
        command = [sys.argv[1]]

    if len(command) > 0:

        if command[0] == "train":
            pca, clf1 = training.train_pca_svm_model()
            clf2 = training.train_hog_svm_model()

            for i, m in enumerate([pca, clf1, clf2]):
                with open(training.MODEL_PATH[i], "wb") as f:
                    pickle.dump(m, f)

        elif command[0] == "pred":

            pca = pickle.load(open(training.MODEL_PATH[0], "rb"))
            clf1 = pickle.load(open(training.MODEL_PATH[1], "rb"))
            clf2 = pickle.load(open(training.MODEL_PATH[2], "rb"))

            filenames = os.listdir(TEST_PATH)
            count = len(filenames)
            wrong = 0
            for s in filenames:
                print("Start predict "+s)
                t0 = time()
                pred = []

                gray_img, gray_fimg, color_img, color_fimg = data_process(
                    os.path.join(TEST_PATH, s))

                # For pca+svm model
                gray_img = gray_img.reshape((1, -1))
                gray_fimg = gray_fimg.reshape((1, -1))
                gray_img_pca = pca.transform(gray_img)
                gray_fimg_pca = pca.transform(gray_fimg)

                pred.append(clf1.predict(gray_img_pca)[0])
                pred.append(clf1.predict(gray_fimg_pca)[0])

                # For hog+svm model

                color_img_hog = FE.get_HOG_vec([color_img])
                color_fimg_hog = FE.get_HOG_vec([color_fimg])

                pred.append(clf2.predict(color_img_hog)[0])
                pred.append(clf2.predict(color_fimg_hog)[0])

                print("done in {0:.3f}s".format(time() - t0))
                print("{0}: {1}, {2}, {3}, {4}".format(
                    s, pred[0], pred[1], pred[2], pred[3]))

                label = int(s.split("_")[0][1:])
                success = False
                for p in pred:
                    if p == label:
                        success = True
                        break
                if not success:
                    wrong += 1

            print("wrong: {0}/{1}".format(wrong, count))

        else:
            print(help_str)
    else:
        print(help_str)


if __name__ == "__main__":
    main()
