from optparse import OptionError
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from keras.layers import Dense,Conv2D,MaxPooling2D,UpSampling2D
from keras import Sequential, Model, Input
import numpy as np
import matplotlib.pyplot as plt

import os
dirname = os.path.dirname(__file__)
train = os.path.join(dirname, '../../data/mnist_train.csv')
test = os.path.join(dirname, '../../data/mnist_test.csv')

def get_MNIST(n_components, reduction: str = "PCA") -> tuple:
    """
    Function to get the MNIST dataset and perform a PCA to it
    """
    train_csv = pd.read_csv(train)
    test_csv = pd.read_csv(test)
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    train_csv = train_csv[train_csv.label.isin(labels)]
    train_y = np.array(train_csv['label'])
    train_X = np.array(train_csv.drop("label", axis=1))

    test_csv = test_csv[test_csv.label.isin(labels)]
    test_y = np.array(test_csv['label'])
    test_X = np.array(test_csv.drop("label", axis=1))

    if reduction == "PCA":
        # We apply PCA to each the training and the test dataset
        train_X = do_pca(n_components, train_X)
        test_X = do_pca(n_components, test_X)
    elif reduction == "AE":
        train_X, test_X = do_AE(n_components, train_X, test_X)
    else:
        raise OptionError()

    return train_X, train_y, test_X, test_y


def do_pca(n_components: int, data: np.ndarray) -> np.ndarray:
    """
    Function to perform PCA to a given data.

    Args:
        n_components: number of components to which we reduce the data
        data: the info we want to apply PCA to
    """
    X = StandardScaler().fit_transform(data)
    pca = PCA(n_components)
    X_pca = np.array(pca.fit_transform(X))
    return X_pca

def do_AE(encoding_dim: int, train_X: np.ndarray, test_X: np.ndarray) -> np.ndarray:
    input_img = Input(shape=(784,))
    # encoded representation of input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # decoded representation of code 
    decoded = Dense(784, activation='sigmoid')(encoded)
    # Model which take input image and shows decoded images
    autoencoder = Model(input_img, decoded)

    # This model shows encoded images
    encoder = Model(input_img, encoded)
    # Creating a decoder model
    encoded_input = Input(shape=(encoding_dim,))
    # last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    train_X = train_X.astype('float32') / 255.
    test_X = test_X.astype('float32') / 255.

    autoencoder.fit(train_X, train_X,
                epochs=20,
                batch_size=256,
                validation_data=(test_X, test_X))

    
    return encoder.predict(train_X), encoder.predict(test_X)

    """
    decoded_img = decoder.predict(encoded_img)
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
        plt.imshow(decoded_img[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    """

def do_AE_CNN(n_components: int, train_X: np.ndarray, test_X: np.ndarray) -> np.ndarray:

    model = Sequential()
    # encoder network
    model.add(Conv2D(30, 3, activation= 'relu', padding='same', input_shape = (28,28,1)))
    model.add(MaxPooling2D(2, padding= 'same'))
    model.add(Conv2D(15, 3, activation= 'relu', padding='same'))
    model.add(MaxPooling2D(2, padding= 'same'))

    #decoder network
    model.add(Conv2D(15, 3, activation= 'relu', padding='same'))
    model.add(UpSampling2D(2))
    model.add(Conv2D(30, 3, activation= 'relu', padding='same'))
    model.add(UpSampling2D(2))
    model.add(Conv2D(1,3,activation='sigmoid', padding= 'same')) # output layer
    model.compile(optimizer= 'adam', loss = 'binary_crossentropy')


    train_X = train_X.astype('float32') / 255.
    test_X = test_X.astype('float32') / 255.
    train_X = np.reshape(train_X, (len(train_X), 28, 28, 1))
    test_X = np.reshape(test_X, (len(test_X), 28, 28, 1))
    model.fit(train_X, train_X,
                    epochs=15,
                    batch_size=128,
                    validation_data=(test_X, test_X))

    pred = model.predict(test_X)

    #create new model
    new_model= Sequential()
    new_model.add(Conv2D(30, 3, activation= 'relu', padding='same', input_shape = (28,28,1)))
    new_model.add(MaxPooling2D(2, padding= 'same'))
    new_model.add(Conv2D(15, 3, activation= 'relu', padding='same'))
    new_model.add(MaxPooling2D(2, padding= 'same'))

    #set weights of the first layer
    new_model.set_weights(model.layers[0].get_weights())

    #compile it after setting the weights
    new_model.compile(optimizer='adam', loss='categorical_crossentropy')

    #get output of the first dens layer
    # output = new_model.predict(samples)

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
        plt.imshow(pred[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()




