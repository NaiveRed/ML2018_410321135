# coding=utf-8
import matplotlib.pyplot as plt
import cv2 as cv


def show_img(img, name="test", fig_n="test_fig", to_RGB=False, gray=False):
    plt.figure(num=fig_n, tight_layout=True)
    if gray:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img if not to_RGB else cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title(name)
    plt.axis("off")
    plt.show()


def show_subplot(row, col, fig_n, imgs, titles):
    _, axes = plt.subplots(row, col, num=fig_n, tight_layout=True)
    for ax, img, title in zip(axes, imgs, titles):
        ax.axis("off")
        ax.imshow(img)
        ax.set_title(title)

    plt.show()
