# Handwritten Character Recognition

## 說明

使用 [MNIST 的資料](http://yann.lecun.com/exdb/mnist/) 來進行手寫辨識的訓練。  

## 環境配置

OS: windows 10  
Python: 3.6.5  
Other packages: numpy, sklearn

## 資料

利用 `mnist = fetch_mldata("MNIST original")` 來取得資料，並存成 .npz 方便下次讀取。
分成以下四組：

    X_train: training image (60000)  
    y_train: training label  
    X_test: testing image (10000)  
    y_test: testing label  

這邊遇到一個 fetch 失敗的問題，解決參考：  

<https://github.com/ageron/handson-ml/issues/143>

> copy https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat file into your scikit data home dir.  
> How to find where is your "scikit data home dir"?
>
>   `from sklearn.datasets.base import get_data_home`  
>   `print (get_data_home())`

## 方法

這邊主要採用 **SVM(support vector machine)**。  
使用 sklearn 中的 `svm.SVC()` 來進行訓練，並搭配其他預處理。

可以利用程式前面的變數來調試：

```
turn_binary = False
range01_normalize = False
mean_normalize = True
do_PCA = True
n_comp = 50
```

以下分別說明各部分的處理：

### Convert to binary color

將原本灰階 [0, 255] 的顏色轉換成 黑(1),白(0)。  
可用 `plt.imshow(img,cmap='binary')` 去顯示看看。

```
X_train[X_train > 0] = 1
X_test[X_test > 0] = 1
```

### Range [0, 1] normalization

將原本 [0, 255] 的值對應到 [0, 1]。

```
X_train /= 255.
X_test /= 255.
```

### Mean normalization

將圖片的所有 pixels 減掉它們的平均數。(x' = x - μ)

```
X_train -= np.mean(X_train)
X_test -= np.mean(X_test)
```

### PCA to Reducing Dimension

使用 PCA(Principal Component Analysis) 將維度減少至特定數量。

```
pca_model = PCA(n_components=n_comp, whiten=True, copy=False)
X_train = pca_model.fit_transform(X_train)
X_test = pca_model.transform(X_test)
```

其中 `n_comp` 是最後每張圖的維度，這邊使用 35 或 50  做嘗試。

### SVM

利用 sklearn 現有的 `svm.SVC` 做訓練。  

使用兩種不同參數，預設的是 `C=1.0, kernel='rbf'`，用來搭配大部分處理。  
其中第二種在將顏色轉為 binary 的情況下會表現較好。

```
1. svm_clf = svm.SVC()
2. svm_clf = svm.SVC(C=7, gamma=0.009)

svm_clf.fit(X_train, y_train)
```

## 結果

以下結果皆使用以上方法和 mnist 的資料所產生。  
過程中使用同樣方法數次，準確度會有些許不同，估計是一些誤差。

其中表現最好的是使用 mean normalization, PCA(n_components=50, whiten=True), SVC(C=1.0, kernel='rbf'):  
testing set acc: 9835/10000(0.9835)

### Only SVM

```
SVC(C=1.0, kernel='rbf'):

    training time: 540.021601 sec
    training set acc: 0.943
    testing set acc: 9446/10000(0.9446)
```

### Convert to binary color

```
convert to binary color,
SVC(C=1.0, kernel='rbf'):

    training time: 471.791367 sec
    training set acc: 0.9493166666666667
    testing set acc: 9483/10000(0.9483)

convert to binary color,
SVC(C=7, gamma=0.009):

    training time: 314.692959 sec
    training set acc: 0.9996666666666667
    testing set acc: 9812/10000(0.9812)
```

### PCA and SVM

先使用 PCA 降維，再進行 SVM 的訓練：

```
PCA(n_components=35, whiten=True),
SVC(C=1.0, kernel='rbf'):

    training time: 28.733455 sec
    training set acc: 0.9922
    testing set acc: 9821/10000(0.9821)

PCA(n_components=50, whiten=True),
SVC(C=1.0, kernel='rbf'):

    training time: 45.877807 sec
    training set acc: 0.9936333333333334
    testing set acc: 9833/10000(0.9833)
```

### Normalization

再進行 PCA 之前先做 **mean normalization**：

```
mean normalization,
PCA(n_components=35, whiten=True),
SVC(C=1.0, kernel='rbf'):

    training time: 27.359246 sec
    training set acc: 0.9922166666666666
    testing set acc: 9823/10000(0.9823)

mean normalization,
PCA(n_components=50, whiten=True),
SVC(C=1.0, kernel='rbf'):

    training time: 49.567197 sec
    training set acc: 0.9937166666666667
    testing set acc: 9835/10000(0.9835)
```

在一開始先將 [0, 255] 對應到 [0, 1]：

```
[0,255] map to [0,1],
mean normalization,
PCA(n_components=35, whiten=True),
SVC(C=1.0, kernel='rbf'):

    training time: 27.148366 sec
    training set acc: 0.9922666666666666
    testing set acc: 9825/10000(0.9825)

[0,255] map to [0,1],
mean normalization,
PCA(n_components=50, whiten=True),
SVC(C=1.0, kernel='rbf'):

    training time: 42.878326 sec
    training set acc: 0.9936166666666667
    testing set acc: 9835/10000(0.9835)
```

## Reference

[A Beginner's Approach to Classification(classify digits)](https://www.kaggle.com/archaeocharlie/a-beginner-s-approach-to-classification?scriptVersionId=470167)  
[sklearn-svm](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)  
[PCA+SVM](https://www.kaggle.com/zhanghuahua/pca-svm)  

SVM 相關課程： [MIT OpenCourseWare](https://www.youtube.com/watch?v=_PwhiWxHK8o)

PCA:  
[StatQuest: Principal Component Analysis (PCA)](https://youtu.be/FgakZw6K1QQ)  
[StatQuest: Principal Component Analysis (PCA)(2015)](https://youtu.be/_UVHneBUBW0)  
