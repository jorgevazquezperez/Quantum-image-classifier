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

def get_MNIST(n_components, reduction: str = "PCA", labels: np.ndarray = [0,1,2,3,4,5,6,7,8,9], visualize: bool = False) -> tuple:
    """
    Function to get the MNIST dataset and perform a dimension reduction to it.
    To perform the dimension reduction we can use:
        - PCA
        - Simple autoencoder
        - Autoencoder based on CNN.

    Args:
        n_components: number of components to which we reduce the data
        reduction: type of reduction we will use
        labels: labels that will be read of the whole dataset
        visualize: bool determining wheter to save the reconstruction of the
        points or not
    Raise:
        OptionError: if the reduction method is not implemented
    """
    mnist = tk.datasets.mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    # Filter the dataset by labels
    if not labels == [0,1,2,3,4,5,6,7,8,9]:
        train_mask = np.isin(train_y, labels)
        test_mask = np.isin(test_y, labels)
        train_X, train_y = train_X[train_mask], train_y[train_mask]
        test_X, test_y = test_X[test_mask], test_y[test_mask]

    # Perform the selected method (or None)
    if reduction == "PCA":
        train_X, test_X = _do_PCA(n_components, train_X, test_X, visualize)
    elif reduction == "AE":
        train_X, test_X = _do_AE(n_components, train_X, test_X, visualize)
    elif reduction == "AE_CNN":
        train_X, test_X = _do_AE_CNN(n_components, train_X, test_X, visualize)
    elif reduction == "None":
        train_X = train_X.reshape((len(train_X), 784))
        test_X = test_X.reshape((len(test_X), 784))
        return train_X, train_y, test_X, test_y
    else:
        raise OptionError()

    return train_X, train_y, test_X, test_y


def _do_PCA(n_components: int, train_X: np.ndarray, test_X: np.ndarray, visualize: bool = False) -> tuple:
    """
    Function to perform PCA to a given data.

    Args:
        n_components: number of components to which we reduce the data
        train_X: the train set we want to apply PCA to
        test_X: the test set we want to apply PCA to
        visualize: bool determining wheter to save the reconstruction of the
        points or not
    Returns:
        tuple: with the train and test arrays preprocessed
    """

    # We reshape and standarize the data
    train_X = train_X.reshape((len(train_X), 784))
    test_X = test_X.reshape((len(test_X), 784))
    test_X_original = test_X[:]

    train_X = StandardScaler().fit_transform(train_X)
    test_X = StandardScaler().fit_transform(test_X)
    pca = PCA(n_components)

    # We perform the PCA
    train_X = pca.fit_transform(train_X)
    test_X = pca.fit_transform(test_X)

    if visualize is True:
        reconstruction = pca.inverse_transform(test_X)
        _visualize_MNIST_pred(test_X_original, reconstruction, "./pca_results.png")

    # Substract by the minimum to make all data positive without
    # modifying the space
    train_X = np.array(train_X) - train_X.min()
    test_X = np.array(test_X) - test_X.min()
    return train_X, test_X

def _do_AE(encoding_dim: int, train_X: np.ndarray, test_X: np.ndarray, visualize: bool = False) -> tuple:
    """
    Function to apply a simple autoencoder to a given data.

    Args:
        n_components: number of components to which we reduce the data
        train_X: the train set we want to apply AE to
        test_X: the test set we want to apply AE to
        visualize: bool determining wheter to save the reconstruction of the
        points or not
    Returns:
        tuple: with the train and test arrays preprocessed
    """
    train_X = train_X.reshape((len(train_X), 784))
    test_X = test_X.reshape((len(test_X), 784))

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

    if visualize is True:
        # Creating a decoder model
        encoded_input = Input(shape=(encoding_dim,))
        # last layer of the autoencoder model
        decoder_layer = autoencoder.layers[-1]
        # decoder model
        decoder = Model(encoded_input, decoder_layer(encoded_input))
        encoded_img = encoder.predict(test_X)
        decoded_img = decoder.predict(encoded_img)
        _visualize_MNIST_pred(test_X, decoded_img, "./ae_simple_results.png")

    train_X = encoder.predict(train_X)
    test_X = encoder.predict(test_X)

    # Substract by the minimum to make all data positive without
    # modifying the space
    train_X = np.array(train_X) - train_X.min()
    test_X = np.array(test_X) - test_X.min()
    return train_X, test_X

def _do_AE_CNN(n_components: int, train_X: np.ndarray, test_X: np.ndarray, visualize: bool = False) -> tuple:
    """
    Function to apply a autoencoder based on a CNN to a given data.

    Args:
        n_components: number of components to which we reduce the data
        train_X: the train set we want to apply AE based on CNN to
        test_X: the test set we want to apply AE based on CNN to
        visualize: bool determining wheter to save the reconstruction of the
        points or not
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

    train_X = train_X.reshape(len(train_X), 28, 28, 1)
    test_X = test_X.reshape(len(test_X), 28, 28, 1)
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

    if visualize is True:
        _visualize_MNIST_pred(test_X, model.predict(test_X), "./ae_cnn_results.png")
    
    train_X = new_model.predict(train_X)
    test_X = new_model.predict(test_X)

    # Substract by the minimum to make all data positive without
    # modifying the space
    train_X = np.array(train_X) - train_X.min()
    test_X = np.array(test_X) - test_X.min()
    return train_X, test_X

def _visualize_MNIST_pred(test_X: np.ndarray, prediction: np.ndarray, name: str):
    """ Function to show the first 5 reconstructions """

    plt.figure(figsize=(20, 4))
    for i in range(5):
        
        # Display original
        ax = plt.subplot(2, 5, i + 1)
        plt.imshow(test_X[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, 5, i + 1 + 5)
        plt.imshow(prediction[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(name)
    plt.show()

