# -*- coding: utf-8 -*-

# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder

import numpy as np

# Specify path to files
train_set_path = 'train-images.idx3-ubyte'
train_labels_path = 'train-labels.idx1-ubyte'
test_set_path = 't10k-images.idx3-ubyte'
test_labels_path = 't10k-labels.idx1-ubyte'

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (28, 28, 1), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 10, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Generator to iterate over the dataset
def read_minst_images(img_path, lbl_path, batch_size = 100):
    # Open buffers to binary files
    img_f = open(img_path, 'rb')
    lbl_f = open(lbl_path, 'rb')
    
    # prepare the onehot encoder to encode the values 0-9 as a 10 element vector
    onehotencoder = OneHotEncoder(categorical_features= [0])
    classes = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
    onehotencoder.fit(classes)
    
    # read first lines of metadata
    magic_number = int.from_bytes(img_f.read(4), byteorder = 'big', signed = False)
    num_of_images = int.from_bytes(img_f.read(4), byteorder = 'big', signed = False)
    num_of_rows = int.from_bytes(img_f.read(4), byteorder = 'big', signed = False)    
    num_of_cols = int.from_bytes(img_f.read(4), byteorder = 'big', signed = False)    
    magic_number_labels = int.from_bytes(lbl_f.read(4), byteorder = 'big', signed = False)
    num_of_labels = int.from_bytes(lbl_f.read(4), byteorder = 'big')
    
    inputs = []
    targets = []
    batch_ctr = 0
    while True:
        for img in range(num_of_images):        
            # Read the image px by px, assigning the value to a matrix of rows & cols
            mat = np.empty((num_of_rows, num_of_cols), dtype = np.uint8)
            for i in range(0,num_of_rows):
                for j in range(0, num_of_cols):
                    mat[i][j] = int.from_bytes(img_f.read(1), byteorder = 'big')
            
            inputs.append(mat)
            targets.append([int.from_bytes(lbl_f.read(1), byteorder = 'big', signed = False)])
            batch_ctr += 1
            # When the input reaches the batch size, a tuple of imgs & labels is built
            if(batch_ctr >= batch_size):
                batch_ctr = 0;
                # build a numpy array based on inputs.
                # Notice that the X element needs to have 4 dims (channels, x, y)
                # Even if the image is grayscale (1 channel) the channel needs to be explicitly defined
                X = np.expand_dims(np.array(inputs),3)
                # Encode the label as a 10 element vector with the value at the index of the category is set to 1
                y = onehotencoder.transform(np.array(targets)).toarray()
                inputs.clear() 
                targets.clear()
                yield X, y
        
        # When the file is over, reset the buffers to bytes 16 in the images file, and 8 in the labels file
        # This is necessary to offset the metadata at the beginning of each file
        img_f.seek(16)
        lbl_f.seek(8)
            
# Build Generator to training dataset
training_set_gen = read_minst_images(train_set_path, train_labels_path)

# Build Generator to test set
test_set_gen = read_minst_images(test_set_path, test_labels_path)

# Fit the classifier
classifier.fit_generator(training_set_gen,
                         samples_per_epoch = 600,
                         nb_epoch = 10,
                         validation_data = test_set_gen,
                         nb_val_samples = 100)