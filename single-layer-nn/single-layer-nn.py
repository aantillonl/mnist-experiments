# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:14:40 2017

@author: aantillonl
"""
import numpy as np

# Specify path to files
train_set_path = 'train-images.idx3-ubyte'
train_labels_path = 'train-labels.idx1-ubyte'
test_set_path = 't10k-images.idx3-ubyte'
test_labels_path = 't10k-labels.idx1-ubyte'

# Read the training set images
with open(train_set_path, "rb") as f:
    magic_number = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    num_of_images = int.from_bytes(f.read(4), byteorder = 'big', signed = False)

    num_of_rows = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    num_of_cols = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    X_train = np.empty((num_of_images, num_of_rows, num_of_cols), dtype = np.uint8)
    
    for img in range(num_of_images):
        for i in range(0,num_of_rows):
            for j in range(0, num_of_cols):
                X_train[img][i][j] = int.from_bytes(f.read(1), byteorder = 'big')

# Getting training set labels
with open(train_labels_path, "rb") as f:
    magic_number_train_labels = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    num_of_labels = int.from_bytes(f.read(4), byteorder = 'big')
    
    y_train = np.empty((num_of_labels), dtype = np.uint8)
    
    for i in range(num_of_images):
        y_train[i] = int.from_bytes(f.read(1), byteorder = 'big', signed = False)


# Getting test images
with open(test_set_path, "rb") as f:
    magic_number = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    num_of_images = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    num_of_rows = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    num_of_cols = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    X_test = np.empty((num_of_images, num_of_rows, num_of_cols), dtype = np.uint8)
    
    for img in range(num_of_images):
        for i in range(0,num_of_rows):
            for j in range(0, num_of_cols):
                X_test[img][i][j] = int.from_bytes(f.read(1), byteorder = 'big', signed = False)
                
# Getting test set labels
with open(test_labels_path, "rb") as f:
    magic_number_train_labels = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    num_of_labels = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    y_test = np.empty((num_of_labels), dtype = np.int8)
    
    for i in range(num_of_images):
        y_test[i] = int.from_bytes(f.read(1), byteorder = 'big', signed = False)


# Encoding labels to a vector of 10 elements
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb.fit([x for x in range (0,10)])
y_train = lb.transform(y_train)
y_test = lb.transform(y_test)

# Binarizing images with a threshold of 120
X_train[:,:,:] = X_train[:,:,:] > 120
X_test[:,:,:] = X_test[:,:,:] > 120

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten

# Initialising the ANN
classifier = Sequential()

# Flattening the input 
classifier.add(Flatten(input_shape = (28,28)))

# Adding the single processing layer
classifier.add(Dense(output_dim = 10, activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 10)

# Part 3 - Making the predictions and evaluating the model
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Get the max element from the probabilities vector. y_test is basicalli decoded back to the labels
y_pred = np.argmax(y_pred, axis = 1)
y_test = np.argmax(y_test, axis = 1)

# Compute the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)