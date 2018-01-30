# Convolutional neural network classifier

This Python script builds and trains a convolutional neural network with the data from the MINST dataset.

The convolutional neural network achieves an accuracy of 99.8% after 10 epochs of training.

The building blocks of the network as the following:

1. **Convolution layer**: Defines the number of filters and the size of the filter kernel. The size of the input is also indicated in this layer.
2. **Pooling**: Reduces the number of features taking the largest values on a 2x2 mask
3. **Secondary convolution layer**
4. **Secondary pooling**
5. **Flattening layer**: Reduces the dimensionality of the input from 2D to a vector of features
6. **Dense layer**: A fully connected layer
7. **Output layer**: The output layer has 10 neurons (the number of classes)

## Reading the data with a generator

Since the size of the input is considerably large, a generator is used to read the input without loading the entire file on memory first.

The `read_minst_images` generator takes as parameters the path to the images and labels files, and a batch size parameter(`img_path, lbl_path, batch_size`).

The generator yelds a batch of size `batch_size` everytime the `next(generator)` function is called. The batch consists of a tuple **X,y** where **X** is a vector of images, and **y** is a vector categories.

### About the images

The images vector **X** must contain 3D images where the dimensions are *height*, *width* and *channels*, channels are the *RGB* values for a color image, even if the image is in gray scale the images should explicitely have 1 channel. That is why in this generator uses the numpy method `np.expand_dims` to add one dimension to the images.

### About the labels

Labels must be encoded in order to be processed and compared accross the neural network. Since we know the MNIST dataset consist of images of handwritten single digit numbers, we also know that there are a total of 10 classes for the data sets (numbers 0 through 9).

The encoding process consists of mapping each label, to a vector with the size equal to the number of classes (10), which has 1, and only 1, element set to true, and all others to false, the element set to 1 is always the same for the same class (e.g. the zero-th element for all cases where the digit in the image is 0).

The encoding is done using the **OneHotEncoder** class from the **sklearn** library.