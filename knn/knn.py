# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:14:40 2017

@author: aantillonl
"""
import numpy as np
import cv2

#import matplotlib.pyplot as plt


# Specify path to files
train_set_path = 'train-images.idx3-ubyte'
train_labels_path = 'train-labels.idx1-ubyte'
test_set_path = 't10k-images.idx3-ubyte'
test_labels_path = 't10k-labels.idx1-ubyte'

# Read the training set images
with open(train_set_path, "rb") as f:
    magic_number = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    num_of_train_images = int.from_bytes(f.read(4), byteorder = 'big', signed = False)

    num_of_rows = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    num_of_cols = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    train_images = np.empty((num_of_train_images, num_of_rows, num_of_cols), dtype = np.uint8)
    
    for img in range(num_of_train_images):
        for i in range(0,num_of_rows):
            for j in range(0, num_of_cols):
                train_images[img][i][j] = int.from_bytes(f.read(1), byteorder = 'big')

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
    
    test_images = np.empty((num_of_test_images, num_of_rows, num_of_cols), dtype = np.uint8)
    
    for img in range(num_of_test_images):
        for i in range(0,num_of_rows):
            for j in range(0, num_of_cols):
                test_images[img][i][j] = int.from_bytes(f.read(1), byteorder = 'big', signed = False)
                
# Getting test set labels
with open(test_labels_path, "rb") as f:
    magic_number_train_labels = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    num_of_test_labels = int.from_bytes(f.read(4), byteorder = 'big', signed = False)
    
    y_test = np.empty((num_of_test_labels), dtype = np.int8)
    
    for i in range(num_of_test_labels):
        y_test[i] = int.from_bytes(f.read(1), byteorder = 'big', signed = False)

# Binarizing images with a threshold of 120
train_images[:,:,:] = train_images[:,:,:] > 120
test_images[:,:,:] = test_images[:,:,:] > 120

# Feature 1. Number of long lines found in the image
long_lines_train = np.empty((num_of_train_images))
for i in range(num_of_train_images):
    # Compute lines longer than 15 px, with a tolerance to gaps of 1 px
    lines = cv2.HoughLinesP(train_images[i],0.5,np.pi/360,5,0,15,1)
    long_lines_train[i] = len(lines) if lines is not None else 0
    
long_lines_test = np.empty((num_of_test_images))
for i in range(num_of_test_images):
    lines = cv2.HoughLinesP(test_images[i],0.5,np.pi/360,5,0,15,1)
    long_lines_test[i] = len(lines) if lines is not None else 0    

# Feature 2: Zoning and pixel density by zone
# Divide the image in zones and Pixel density in 16 zones (4x4)
num_zones = 16
zone_vectors_train = np.empty((num_of_train_images, num_zones), dtype = np.uint8)
for k in range(len(train_images)):
    zone_counter = 0
    for i in range(0,4):
        for j in range(0,4):
            x_start = i * 7
            x_end = x_start + 7
            y_start = j * 7
            y_end = y_start + 7
            zone_vectors_train[k][zone_counter] = sum(sum(row) for row in train_images[k,y_start:y_end,x_start:x_end])
            zone_counter = zone_counter + 1

zone_vectors_test = np.empty((num_of_test_images, num_zones), dtype = np.uint8)
for k in range(len(test_images)):
    zone_counter = 0
    for i in range(0,4):
        for j in range(0,4):
            x_start = i * 7
            x_end = x_start + 7
            y_start = j * 7
            y_end = y_start + 7
            zone_vectors_test[k][zone_counter] = sum(sum(row) for row in test_images[k,y_start:y_end,x_start:x_end])
            zone_counter = zone_counter + 1

# Merge feature vectors
X_train = np.empty((num_of_train_images,17))
X_train[:,0:16] = zone_vectors_train
X_train[:,16] = long_lines_train

X_test = np.empty((num_of_test_images,17))
X_test[:,0:16] = zone_vectors_test
X_test[:,16] = long_lines_test


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

overall_accuracy = sum(cm[i,i] for i in range(len(cm)))/num_of_test_images
