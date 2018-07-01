# Face Recognizer

使用所提供的人臉圖片進行辨識訓練並做測試。

## Usage

```
Usage:

    Train the model:
        python face_recognizer.py train

    Predict the image in data/test_img:
        python face_recognizer.py pred
```

## Training Data

訓練資料共 650 張，檔名為 `sxx_oo.jpg`，放置在 `data\face_database`。  
e.g. `s08_oo`: label 為 08。

## Program

`training.py`:  訓練相關及資料標記。  
`feature_extractor.py`: 抽取訓練所需的特徵。  
`utility.py`: 顯示測試。  
`face_recognizer.py`: 將訓練及預測資料的處理統整在一起，最後的操作介面。

首先將檔案標記好對應 label 並轉換成灰階或 BGR 且統一大小。  
這邊為了使訓練資料更多，我們將每張圖都水平翻轉，這樣總共就有 1300 張。  
測試上使用 0.1 的資料量(130 張未在訓練中)。

訓練則分別使用兩種方法： PCA + SVM、HOG + SVM

## Method

### PCA + SVM

先將圖片以灰階方式讀入(統一至平均大小)，接著進行 PCA 將 feature vectors 抽取出來(160 dim)。   
(`PCA(n_components=PCA_N, svd_solver='randomized',whiten=True).fit(X)`)

把 feature vectors 丟入 SVM 中訓練，這裡使用 `SVC(kernel='rbf', class_weight='balanced')`。  
並利用 `GridSearchCV` 找出給定的較佳參數。  
(`param_grid = {'C': [1, 1e3, 5e3, 1e4],'gamma': [0.0001, 0.0005, 0.001, 0.005]}`)

以 0.1 的資料去做測試：正確率約為 0.85。

相關訓練及測試的 log：  
```
========== Start construct PCA+SVM model ==========
Generate labeled data...
total image: 1300
total label: 1300
done in 0.946s
Compute PCA...
done in 6.606s
Use PCA on data...
done in 0.680s
shape of PCA feature vectors: (1170, 160)

Fitting the classifier(PCA+SVM) to the training set..
done in 30.056s
Best estimator found by grid search:
SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.005, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Training set acc: 1.0

Testing set:
shape of feature vectors: (130, 160)
Predicting on the testing set...
done in 0.052s
testing set acc: 0.8538461538461538
=============== DONE! ===============
```

### HOG + SVM

先將圖片以彩色方式(BGR)讀入(統一至平均大小)，接著利用 skimage 的 `hog()` 取出 HOG features vector，
相關參數：  
```
ORIENT = 9  # number of bins for HOG
PIXELS_PER_CELL = (32, 32)  # 32x32
CELLS_PER_BLOCK = (2, 2)  # 2x2 cell
BLOCK_NORM = "L2"
```

再來利用 linear SVM: `LinearSVC(class_weight='balanced')` 對其進行訓練和找出較佳的參數。  

以 0.1 的資料去做測試：正確率約為 0.83。

相關訓練及測試的 log：  
```
========== Start construct HOG+SVM model ==========
Generate labeled data...
total image: 1300
total label: 1300
image shape: 142 x 208 x 3 (W x H x COLOR)
done in 1.232s
shape of HOG feature vectors: (1170, 540)

Fitting the classifier(HOG+SVM) to the training set..
done in 48.171s
Best estimator found by grid search:
LinearSVC(C=1, class_weight='balanced', dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
Training set acc: 1.0

Testing set:
shape of HOG feature vectors: (130, 540)
Predicting on the testing set...
done in 0.004s
testing set acc: 0.8307692307692308
=============== DONE! ===============
```

### Recognizer

將所有圖片(1300 張)去做訓練後產生 model，並用來做預測。  

預測時會將圖片以灰階和彩色的方式讀入，接著各自進行水平反轉，這樣輸入就變為四張。  
再把它們分別丟進兩個分類器，輸出四個結果。(灰階的丟 PCA、彩色的丟 HOG。)  
而為符合題目要求，這邊只要四個結果有一個答對，及算為正確。

## Results

[Demo log](demo_pred.txt)  
使用全部資料進行訓練(1300 張)，並以 100 張(助教提供)未在訓練集中的資料來測試，正確率為 92 %。

## Reference

[Face Recognition](http://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html)  
[HOG](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html)