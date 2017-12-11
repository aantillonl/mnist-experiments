# -*- coding: utf-8 -*-
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

# Read the training set images. Images are read as a single vector, not as 2d arrays
with open(train_set_path, "rb") as f:
    magic_number = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    num_of_train_images = int.from_bytes(f.read(4), byteorder = 'big', signed = False)

    num_of_rows = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    num_of_cols = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    X_train = np.empty((num_of_train_images, num_of_rows * num_of_cols), dtype = np.uint8)
    
    for i in range(num_of_train_images):
        for j in range(0,num_of_rows * num_of_cols):
            X_train[i][j] = int.from_bytes(f.read(1), byteorder = 'big')

# Getting training set labels
with open(train_labels_path, "rb") as f:
    magic_number_train_labels = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    num_of_train_labels = int.from_bytes(f.read(4), byteorder = 'big')

    y_train = np.empty((num_of_train_labels), dtype = np.uint8)
    
    for i in range(num_of_train_labels):
        y_train[i] = int.from_bytes(f.read(1), byteorder = 'big', signed = False)


# Getting test images
with open(test_set_path, "rb") as f:
    magic_number = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    num_of_test_images = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    num_of_rows = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    num_of_cols = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    X_test = np.empty((num_of_test_images, num_of_rows*num_of_cols), dtype = np.uint8)
    
    for i in range(num_of_test_images):
        for j in range(0,num_of_rows*num_of_cols):
            X_test[i][j] = int.from_bytes(f.read(1), byteorder = 'big', signed = False)
                
# Getting test set labels
with open(test_labels_path, "rb") as f:
    magic_number_train_labels = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    num_of_test_labels = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    y_test = np.empty((num_of_test_labels), dtype = np.int8)
    
    for i in range(num_of_test_labels):
        y_test[i] = int.from_bytes(f.read(1), byteorder = 'big', signed = False)
    
# Extract principal components. Top 50 components
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
zone_vectors_train = sc.fit_transform(X_train)
zone_vectors_test = sc.transform(X_test)

# KNN classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predict
y_pred = classifier.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

general_accuracy = sum(cm[i,i] for i in range(len(cm)))/num_of_test_images
