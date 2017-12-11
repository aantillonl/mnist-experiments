# KNN classifier

This section compares 2 variations of KNN classifiers.

The first classifier uses two feature extraction techniques that were manually chosen and implemented on the algorithm, while the second classifier uses **Principal Component Analysis** (PCA) to derive additional features from the original dataset that both represent well the variance and reduce the dimension of the original dataset.

## First KNN: Statistical and structural features

Feature extraction techniques are sometimes classified into *statistical* and *structural*. For the first classifiera one statistical and one structural feature were chosen.

The statistical feature used is *zoning*, dividing the picture into 16 zones, a 4x4 grid, and computing the pixel density on each of the zones

The structural feature is finding "long" lines using the *Hough transform* in python library *opencv*. The number of lines of at least 15 px is used as a feature for the KNN classifier.

## Second KNN: PCA

The second classifier uses principal conponent analysis to extract new components that describe well our dataset of images, but are fewer than just using each pixel as a feature. The number of components selected was 50, simply because it lead to good results

## Results

The **Statustical and structural features** method yielded an 82% accuracy in with the MNIST test images. While the **PCA** yielded a 97% accuracy.

## Confusion matrix of PCA KNN

|     | 0 | 1  | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|-----|---|--- |----|---|---|---|---|---|---|---|
|**0**|972| 1  | 1  | 0 | 0 | 1 | 4 | 1 | 0 | 0 |
|**1**| 0 |1131| 2  | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
|**2**| 8 | 0  |1002| 0 | 1 | 0 | 2 | 15| 3 | 0 |
|**3**| 0 | 0  | 1  |977| 1 | 12| 0 | 6 | 7 | 6 |
|**4**| 2 | 4  | 0  | 0 |953| 0 | 4 | 1 | 0 | 18|
|**5**| 3 | 0  | 0  | 5 | 1 |871| 7 | 1 | 2 | 2 |
|**6**| 3 | 4  | 0  | 0 | 3 | 1 |947| 0 | 0 | 0 |
|**7**| 0 | 17 | 5  | 1 | 3 | 0 | 0 |993| 0 | 9 |
|**8**| 5 | 0  | 4  | 10| 2 | 7 | 2 | 2 |937| 5 |
|**9**| 3 | 5  | 3  | 6 | 6 | 5 | 1 | 7 | 4 |969|