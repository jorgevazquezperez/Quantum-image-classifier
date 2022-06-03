from quantum_image_classifier import data_generator, variance_error_graph, data_loader, error_graph, cloud_point
import numpy as np

from sklearn.neighbors import NearestCentroid as classicalNC
from quantum_image_classifier import NearestCentroid as quantumNC

from tqdm import tqdm

def distanceComparison():
    n_dim = 2
    n_clusters = 2
    train_X, train_y, test_X, test_y = data_generator.generate_synthetic_data(
        n_dim, n_clusters, 1000)
    cloud_point(train_X, train_y, "./cloud_point_gen.png")

    quantum = quantumNC()
    quantum.fit(train_X, train_y, n_dim)
    labels, differenceGen, distanceGen = quantum.predict(test_X, test_distance=True)

    train_X, train_y, test_X, test_y = data_loader.get_MNIST(n_dim, "PCA", [0,1])
    cloud_point(train_X, train_y, "./could_point_MNIST.png")

    quantum = quantumNC()
    quantum.fit(train_X, train_y, n_dim)
    labels, differenceMNIST, distanceMNIST = quantum.predict(test_X, test_distance=True)

    print("Generado: MEDIA {} DIFERENCIA {}".format(np.mean(distanceGen), np.mean(differenceGen)))
    print("MNIST: MEDIA {} DIFERENCIA {}".format(np.mean(distanceMNIST), np.mean(differenceMNIST)))


def quantumVSclasico_gen():
    n_dim = 8
    n_clusters = 2
    accuracyQ = []
    accuracyC = []

    for _ in range(1):
        train_X, train_y, test_X, test_y = data_generator.generate_synthetic_data(
            n_dim, n_clusters, 1000)

        quantum = quantumNC()
        quantum.fit(train_X, train_y, n_dim)
        labelsQ = quantum.predict(test_X)

        classical = classicalNC()
        classical.fit(train_X, train_y)
        labelsC = classical.predict(test_X)

        accuracyQ.append(_calc_accuracy(test_y, labelsQ))
        accuracyC.append(_calc_accuracy(test_y, labelsC))

    variance_error_graph(test_y, "./clasico_vs_cuantico_gen.png",
                         ("NC clasico", accuracyC), ("NC cuántico", accuracyQ))

def quantumVSclassic_MNIST():
    n_dim = 8
    n_clusters = 2
    accuracyQ = []
    accuracyC = []

    for _ in range(1):
        train_X, train_y, test_X, test_y = data_loader.get_MNIST(n_dim, "PCA", [
                                                                 1, 0, 6, 9])

        quantum = quantumNC()
        quantum.fit(train_X, train_y, n_dim)
        labelsQ = quantum.predict(test_X)

        classical = classicalNC()
        classical.fit(train_X, train_y)
        labelsC = classical.predict(test_X)

        accuracyQ.append(_calc_accuracy(test_y, labelsQ))
        accuracyC.append(_calc_accuracy(test_y, labelsC))

    variance_error_graph(test_y, "./clasico_vs_cuantico_MNIST.png",
                         ("NC clasico", accuracyC), ("NC cuántico", accuracyQ))

def _calc_accuracy(labels, predictions):
    counts = 0
    for label, prediction in zip(labels, predictions):
        if prediction == label:
            counts += 1
    return counts / len(labels)