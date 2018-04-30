import os
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt

DEBUG = 0

help = """
Usage:
  python main.py gen
  python main.py dec <relative path of image>
  python main.py enc <relative path of image>
Command:
  gen\t\tGenerate weight vector
  dec\t\tDecrypt the image(need weight)
  enc\t\tEncrypt the image(need weight)
"""

data_path = os.path.join(os.getcwd(), "ML_data")
k1 = cv2.imread(os.path.join(data_path, "key1.png"), cv2.IMREAD_GRAYSCALE)
k2 = cv2.imread(os.path.join(data_path, "key2.png"), cv2.IMREAD_GRAYSCALE)

fig = plt.figure(num="Result Image", tight_layout=True)


def PLA(I, E, k1, k2):
    '''
    Perceptron learning algorithm:

    a = w(k).T * x (dot product)
    error = E - a
    w(k+1) = w(k) + x * lr * error
    '''

    max_epo = 5  # maximal number of epochs
    epsilon = 0.1  # check the convergence of weight vectors (兩次更新之間的差距)
    lr = 0.00001  # learning rate
    epo = 1

    W = np.random.randn(3, 1)
    W_last = None
    # W = np.zeros(3, 1)

    height, width = I.shape
    while epo == 1 or (epo <= max_epo and np.linalg.norm(W-W_last) > epsilon):
        print("Epoch:", epo)
        W_last = np.copy(W)
        for px in range(height):
            for py in range(width):
                x = np.array([[k1[px, py]], [k2[px, py]], [I[px, py]]])
                a = W.T.dot(x)
                error = E[px, py]-a[0]
                W += lr*error*x
        print("Epsilon:", np.linalg.norm(W-W_last))
        epo += 1

    print("Weights:", W, sep='\n')
    return W


def normalize(x):
    '''
    normalize the value to [0, 255](gray scale)
    x' = (x - min) * (255 - 0)/(max - min) + 0
    '''
    min_x = x.min()
    ratio = 255/(x.max()-min_x)
    height, width = x.shape
    new_x = np.zeros(x.shape, dtype=x.dtype)

    for px in range(height):
        for py in range(width):
            new_x[px, py] = int(round((x[px, py]-min_x)*ratio))

    return new_x


def decrypt(E, W, k1, k2):
    '''
    Use W(weights) and key image(k1, k2) to decrypt E
    I = (E - w0k1 - w1k2) / w2
    '''

    height, width = E.shape
    I = [(E[px, py]-W[0]*k1[px, py]-W[1]*k2[px, py])/W[2]
         for px in range(height)for py in range(width)]

    if DEBUG == 2:
        I = np.array(I)
        print("range:[", I.min(), I.max(), "]")
        print("type:", I.dtype)

    I = normalize(np.array(I))  # 先將值對應到 [0, 255] 的整數
    I = I.astype(np.uint8).reshape(E.shape)  # 轉成 8-int unsigned

    '''
    不處理 float 和超過的範圍，cv2.imshow 顯示錯誤，
    但可在外部顯示正確，估計是直接對超過 0/255 的當成 0(黑)/255(白) 來處理。

    正確的 float 範圍應在 [0,1]，因 imshow 會將 float 直接乘 255 來 map 到 [0,255]。
    而我們的值並不在 [0,1]，乘完後都超過了 255。
    I = np.array(I).reshape(E.shape)
    '''

    '''
    直接將其轉成 8-bit unsigned 會出錯
    I = np.array(I)
    I = I.astype(np.uint8).reshape(E.shape)
    '''
    return I


def encrypt(I, W, k1, k2):
    '''
    Use W(weights) and key image(k1, k2) to encrypt I
    E = w0k1 + w1k2 + w2I
    '''

    height, width = k1.shape
    if k1.shape != I.shape:
        I = cv2.resize(I, (width, height), cv2.INTER_LINEAR)

    E = [W[0]*k1[px, py]+W[1]*k2[px, py]+W[2]*I[px, py]
         for px in range(height)for py in range(width)]

    E = normalize(np.array(E))
    E = E.astype(np.uint8).reshape(I.shape)
    return E


def set_image(I, p, name):
    fig.add_subplot(1, 2, p)
    plt.imshow(I, cmap="gray")
    plt.title(name)
    plt.xticks([])
    plt.yticks([])


def main():

    if DEBUG:
        # Read image data
        I = cv2.imread(data_path + "\\I.png", cv2.IMREAD_GRAYSCALE)
        E = cv2.imread(data_path + "\\E.png", cv2.IMREAD_GRAYSCALE)
        cv2.imshow("I", I)
        cv2.imshow("E", E)
        cv2.imshow("k1", k1)
        cv2.imshow("k2", k2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Image: {0}x{1}".format(E.shape[0], E.shape[1]))
        W = PLA(I, E, k1, k2)
        # W.dump(os.path.join(data_path, "weights.dat"))
        # W = np.load(os.path.join(data_path, "weights.dat"))

        # encrypt
        I = cv2.imread(os.path.join(
            data_path, "I.png"), cv2.IMREAD_GRAYSCALE)
        E = encrypt(I, W, k1, k2)
        cv2.imshow("Origin", I)
        cv2.imshow("After encrypted", E)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # decrypt
        # E = cv2.imread(os.path.join(data_path, "E.png"), cv2.IMREAD_GRAYSCALE)
        I = decrypt(E, W, k1, k2)
        cv2.imshow("Origin", E)
        cv2.imshow("After decrypted", I)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif len(sys.argv) > 1:
        if sys.argv[1] == "dec":
            W = np.load(os.path.join(data_path, "weights.dat"))
            E = cv2.imread(os.path.join(
                os.getcwd(), sys.argv[2]), cv2.IMREAD_GRAYSCALE)
            I = decrypt(E, W, k1, k2)

            cv2.imwrite("decrypted.png", I)
            set_image(E, 1, "Origin")
            set_image(I, 2, "After decrypted")
            plt.show()
        elif sys.argv[1] == "enc":
            W = np.load(os.path.join(data_path, "weights.dat"))
            I = cv2.imread(os.path.join(
                os.getcwd(), sys.argv[2]), cv2.IMREAD_GRAYSCALE)
            E = encrypt(I, W, k1, k2)

            cv2.imwrite("encrypted.png", E)
            set_image(I, 1, "Origin")
            set_image(E, 2, "After encrypted")
            plt.show()
        elif sys.argv[1] == "gen":
            I = cv2.imread(os.path.join(data_path, "I.png"),
                           cv2.IMREAD_GRAYSCALE)
            E = cv2.imread(os.path.join(data_path, "E.png"),
                           cv2.IMREAD_GRAYSCALE)
            print("Image: {0}x{1}".format(E.shape[0], E.shape[1]))

            W = PLA(I, E, k1, k2)
            W.dump(os.path.join(data_path, "weights.dat"))

            with open(os.path.join(data_path, "weights.txt"), "w") as f:
                f.write("[{0:.6f} {1:.6f} {2:.6f}]\n".format(
                    W[0, 0], W[1, 0], W[2, 0]))
        else:
            print(help)
    else:
        print(help)


if __name__ == "__main__":
    main()
