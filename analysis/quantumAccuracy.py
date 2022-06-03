from quantum_image_classifier import data_generator, variance_error_graph, data_loader, error_graph, cloud_point
import numpy as np

from sklearn.neighbors import NearestCentroid as classicalNC
from quantum_image_classifier import NearestCentroid as quantumNC

from tqdm import tqdm

def numClustersAccuracy():
    n_dim = 8
    accuracy2 = []
    accuracy4 = []
    accuracy6 = []

    for _ in range(1):
        train_X, train_y, test_X, test_y = data_loader.get_MNIST(n_dim, "PCA", [
                                                                 1, 0])
        quantum = quantumNC()
        quantum.fit(train_X, train_y, n_dim)
        accuracy2.append(_calc_accuracy(test_y, quantum.predict(test_X)))

        train_X, train_y, test_X, test_y = data_loader.get_MNIST(n_dim, "PCA", [
                                                                 1, 0, 6, 9])
        quantum = quantumNC()
        quantum.fit(train_X, train_y, n_dim)
        accuracy4.append(_calc_accuracy(test_y, quantum.predict(test_X)))

        train_X, train_y, test_X, test_y = data_loader.get_MNIST(
            n_dim, "PCA", [1, 0, 4, 5, 6, 9])
        quantum = quantumNC()
        quantum.fit(train_X, train_y, n_dim)
        accuracy6.append(_calc_accuracy(test_y, quantum.predict(test_X)))

        variance_error_graph(test_y, "./clusters_accuracy.png",
                         ("NC clasico", accuracy2), ("NC cuántico", accuracy4), ("NC clasico", accuracy6))

def reductionMethodAccuracy():
    n_dim = 8
    accuracyPCA = []
    accuracyAE = []
    accuracyAECNN = []

    for _ in range(1):
        train_X, train_y, test_X, test_y = data_loader.get_MNIST(n_dim, "PCA", [
                                                                 1, 0, 6, 9])
        quantum = quantumNC()
        quantum.fit(train_X, train_y, n_dim)
        accuracyPCA.append(_calc_accuracy(test_y, quantum.predict(test_X)))

        train_X, train_y, test_X, test_y = data_loader.get_MNIST(n_dim, "AE", [
                                                                 1, 0, 6, 9])
        quantum = quantumNC()
        quantum.fit(train_X, train_y, n_dim)
        accuracyAE.append(_calc_accuracy(test_y, quantum.predict(test_X)))

        train_X, train_y, test_X, test_y = data_loader.get_MNIST(n_dim, "AE_CNN", [
                                                                 1, 0, 6, 9])
        quantum = quantumNC()
        quantum.fit(train_X, train_y, n_dim)
        accuracyAECNN.append(_calc_accuracy(test_y, quantum.predict(test_X)))

        variance_error_graph(test_y, "./reduction_accuracy.png",
                         ("PCA", accuracyPCA), ("AE", accuracyAE), ("AE CNN", accuracyAECNN))

def _calc_accuracy(labels, predictions):
    counts = 0
    for label, prediction in zip(labels, predictions):
        if prediction == label:
            counts += 1
    return counts / len(labels)

