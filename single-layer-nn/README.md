# Single Layer Neural Network

This python script uses a **single layer neural network** to predict the class of a test set of handwritten characters.

## Loading the dataset
The dataset is stored in IDX format (more info in MNIST'S website). Therefore a custom program is needed to read the files, the script opens a buffer to the files, and reads the bytes as specified in the MNIST's website to load the images as 2-d numpy arrays

## Preprocessing
The preprocessing done in this script includes the following steps:

### Encode labels
From a numeric value to a vector of classes with the corresponding class set to 1 and the rest to 0's

### Binarize images
It makes sense to work with binary images of 1's and 0's, rather than the original gray scale values (0-255) because it is only necessary to know whether a pixel is background or it is part of the number. The binarization was done consulting the images histogram to select a suitable threshold, inthis case is 120.

## The newural network

The neural network, if it can be called so since it is only one layer, consists of a group of 10 neurons which are connected to all the pixels on the image.

Each neuron tracks a specific class, and should fire only when the input belong to its class. Since classes are exclusive (i.e. an image can only be one number) ideally for each input only one neuron should fire, while the rest should remain with a low output.

The neurons use a **Softmax** activation function, which is recomended for multiclass problems

The training consisted of 10 epochs with batch of 10 samples.

### The flatten layer

The term "single layer" is written all over this file, nevertheless, there is one layer before the output layer!

However, this extra layer, only *flattens* the input. This means that it makes the input, an image of 28 x 28 px, a single vector of whatever 28 x 28 equals to. This step is necessary because the actual layer that uses the input expects one vector at the time.

## Results

This classifier, regardless its simplicity, showed an accuracy of **92.7 in trining**, and **91.77 in test**.

## Confusion matrix

|     | 0 | 1  | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|-----|---|--- |---|---|---|---|---|---|---|---|
|**0**|961| 0  | 2 | 2 | 0 | 8 | 3 | 2 | 1 | 1 |
|**1**| 0 |1106| 5 | 2 | 0 | 4 | 4 | 1 | 13| 0 |
|**2**| 10| 6  |913| 27| 10| 3 | 12| 4 | 43| 4 |
|**3**| 3 | 0  | 18|923| 2 | 23| 1 | 13| 20| 7 |
|**4**| 3 | 1  | 6 | 3 |908| 1 | 10| 5 | 9 | 36|
|**5**| 11| 3  | 2 | 39| 10|778| 10| 5 | 32| 2 |
|**6**| 12| 3  | 9 | 1 | 11| 25|894| 1 | 2 | 0 |
|**7**|  1| 6  | 27| 8 | 7 | 2 | 0 |932| 1 | 44|
|**8**| 8 | 8  | 8 | 28| 7 | 33| 9 | 9 |854| 10|
|**9**| 12| 4  | 0 | 13| 33| 11| 0 | 22| 6 |908|
