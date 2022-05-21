from optparse import OptionError
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from keras.layers import Dense,Conv2D,MaxPooling2D,UpSampling2D,Flatten, Reshape
from keras import Sequential, Model, Input
import tensorflow.keras as tk
import numpy as np
import matplotlib.pyplot as plt

import os
dirname = os.path.dirname(__file__)
train = os.path.join(dirname, '../../data/mnist_train.csv')
test = os.path.join(dirname, '../../data/mnist_test.csv')

def get_MNIST(n_components, reduction: str = "PCA") -> tuple:
    """
    Function to get the MNIST dataset and perform a dimension reduction to it.
    To perform the dimension reduction we can use:
        - PCA
        - Simple autoencoder
        - Autoencoder based on CNN.

    Args:
        n_components: number of components to which we reduce the data
        reduction: type of reduction we will use
    """
    mnist = tk.datasets.mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    print(type(train_X))

    if reduction == "PCA":
        # We apply PCA to each the training and the test dataset
        train_X, test_X = _do_PCA(n_components, train_X, test_X)
    elif reduction == "AE":
        train_X, test_X = _do_AE(n_components, train_X, test_X)
    elif reduction == "AE_CNN":
        train_X, test_X = _do_AE_CNN(n_components, train_X, test_X)
    else:
        raise OptionError()

    return train_X, train_y, test_X, test_y


def _do_PCA(n_components: int, train_X: np.ndarray, test_X: np.ndarray) -> tuple:
    """
    Function to perform PCA to a given data.

    Args:
        n_components: number of components to which we reduce the data
        data: the info we want to apply PCA to
    Returns:
        tuple: with the train and test arrays preprocessed
    """

    # We reshape and standarize the data
    train_X = train_X.reshape((60000, 784))
    test_X = test_X.reshape((10000, 784))
    train_X = StandardScaler().fit_transform(train_X)
    test_X = StandardScaler().fit_transform(test_X)
    pca = PCA(n_components)

    # We perform the PCA
    train_X = pca.fit_transform(train_X)
    test_X = pca.fit_transform(test_X)

    return train_X, test_X

def _do_AE(encoding_dim: int, train_X: np.ndarray, test_X: np.ndarray) -> tuple:
    """
    Function to apply a simple autoencoder to a given data.

    Args:
        n_components: number of components to which we reduce the data
        data: the info we want to apply the simple autoencoder to
    Returns:
        tuple: with the train and test arrays preprocessed
    """
    train_X = train_X.reshape((60000, 784))
    test_X = test_X.reshape((10000, 784))

    # Autoencoder model creation
    input_img = Input(shape=(784,))
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    decoded = Dense(784, activation='sigmoid')(encoded)
    autoencoder = Model(input_img, decoded)

    # Encoder part constructed
    encoder = Model(input_img, encoded)

    # Compile of the model, scale the data and fit of the model
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    train_X = train_X.astype('float32') / 255.
    test_X = test_X.astype('float32') / 255.
    autoencoder.fit(train_X, train_X,
                epochs=20,
                batch_size=256,
                validation_data=(test_X, test_X))

    
    return encoder.predict(train_X), encoder.predict(test_X)

def _do_AE_CNN(n_components: int, train_X: np.ndarray, test_X: np.ndarray) -> tuple:
    """
    Function to apply a autoencoder based on a CNN to a given data.

    Args:
        n_components: number of components to which we reduce the data
        data: the info we want to apply the autoencoder based on a CNN to
    Returns:
        tuple: with the train and test arrays preprocessed
    """

    model = Sequential()

    # Encoder network
    model.add(Conv2D(30, 3, activation= 'relu', padding='same', input_shape = (28,28,1)))
    model.add(MaxPooling2D(2, padding= 'same'))
    model.add(Conv2D(15, 3, activation= 'relu', padding='same'))
    model.add(MaxPooling2D(2, padding= 'same'))
    model.add(Flatten())
    model.add(Dense(units=n_components, activation="relu"))

    # Decoder network
    model.add(Dense(units=735, activation="relu"))
    model.add(Reshape((7, 7, 15)))
    model.add(Conv2D(15, 3, activation= 'relu', padding='same'))
    model.add(UpSampling2D(2))
    model.add(Conv2D(30, 3, activation= 'relu', padding='same'))
    model.add(UpSampling2D(2))
    model.add(Conv2D(1,3,activation='sigmoid', padding= 'same')) # output layer
    model.compile(optimizer= 'adam', loss = 'binary_crossentropy')

    # Scalate the data and fit the model
    train_X = train_X.astype('float32') / 255.
    test_X = test_X.astype('float32') / 255.
    model.fit(train_X, train_X, epochs=8, batch_size=128, validation_data=(test_X, test_X))
    
    # Create new model to get the data encoded
    new_model= Sequential()
    new_model.add(Conv2D(30, 3, activation= 'relu', padding='same', input_shape = (28,28,1)))
    new_model.add(MaxPooling2D(2, padding= 'same'))
    new_model.add(Conv2D(15, 3, activation= 'relu', padding='same'))
    new_model.add(MaxPooling2D(2, padding= 'same'))
    new_model.add(Flatten())
    new_model.add(Dense(units=n_components, activation="relu"))

    # Set weights of the trained autoencoder to the encoder model
    enc_len = len(model.layers) // 2
    for i, layer in enumerate(model.layers[:enc_len]):
        new_model.layers[i].set_weights(layer.get_weights())

    # Compile it after setting the weights
    new_model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    return new_model.predict(train_X), new_model.predict(test_X)




