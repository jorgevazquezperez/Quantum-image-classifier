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
    print("Primeros elementos DATOS GENERADOS:")
    print(train_X[:10])

    quantum = quantumNC()
    quantum.fit(train_X, train_y, n_dim)
    labels, differenceGen, distanceGen = quantum.predict(test_X, test_distance=True)

    train_X, train_y, test_X, test_y = data_loader.get_MNIST(n_dim, "PCA", [0,1])
    cloud_point(train_X, train_y, "./could_point_MNIST.png")

    quantum = quantumNC()
    quantum.fit(train_X, train_y, n_dim)
    labels, differenceMNIST, distanceMNIST = quantum.predict(test_X, test_distance=True)
    print("Primeros elementos DATOS MNIST:")
    print(train_X[:10])
    
    print("Generado: MEDIA {} DIFERENCIA {}".format(np.mean(distanceGen), np.mean(differenceGen)))
    print("MNIST: MEDIA {} DIFERENCIA {}".format(np.mean(distanceMNIST), np.mean(differenceMNIST)))


def quantumVSclasico_gen():
    n_dim = 8
    n_clusters = 4
    quantum_labels = []
    classical_labels = []

    for _ in tqdm(range(1)):
        train_X, train_y, test_X, test_y = data_generator.generate_synthetic_data(
            n_dim, n_clusters, 1000)

        quantum = quantumNC()
        quantum.fit(train_X, train_y, n_dim)
        quantum_labels.append(quantum.predict(test_X))

        classical = classicalNC()
        classical.fit(train_X, train_y)
        classical_labels.append(classical.predict(test_X))

    variance_error_graph(test_y, "./clasico_vs_cuantico_gen.png",
                         ("NC clasico", classical_labels), ("NC cuántico", quantum_labels))


def quantumVSclassic_MNIST():
    n_dim = 8

    labels_2 = []
    labels_4 = []
    labels_6 = []
    labels = []

    classic_MNIST()
    """
    for _ in tqdm(range(1)):
        train_X, train_y, test_X, test_y = data_loader.get_MNIST(n_dim, "PCA", [
                                                                 1, 0])
        quantum = quantumNC()
        quantum.fit(train_X, train_y, n_dim)
        labels_2.append(quantum.predict(test_X))

        train_X, train_y, test_X, test_y = data_loader.get_MNIST(n_dim, "PCA", [
                                                                 1, 0, 6, 9])
        quantum = quantumNC()
        quantum.fit(train_X, train_y, n_dim)
        labels_4.append(quantum.predict(test_X))

        train_X, train_y, test_X, test_y = data_loader.get_MNIST(
            n_dim, "PCA", [1, 0, 4, 5, 6, 9])
        quantum = quantumNC()
        quantum.fit(train_X, train_y, n_dim)
        labels_6.append(quantum.predict(test_X))


    variance_error_graph(test_y, "./clasico_vs_cuantico_MNIST.png",
                         ("2 clases", labels_2), ("4 clases", labels_4), ("6 clases", labels_6))
    """
    

def classic_MNIST():
    n_dim = 8

    accuracy_2 = []
    accuracy_4 = []
    accuracy_6 = []

    for _ in tqdm(range(1)):
        train_X, train_y, test_X, test_y = data_loader.get_MNIST(n_dim, "AE", [
                                                                 1, 0])
        classical = classicalNC()
        classical.fit(train_X, train_y)
        labels_2 = classical.predict(test_X)
        counts = 0
        for correct, predict in zip(test_y, labels_2):
            if predict == correct:
                counts += 1
        print(counts / len(test_y))

        quantum = quantumNC()
        quantum.fit(train_X, train_y, n_dim)
        labels_2 = quantum.predict(test_X)
        counts = 0
        for correct, predict in zip(test_y, labels_2):
            if predict == correct:
                counts += 1
        print(counts / len(test_y))



        """train_X, train_y, test_X, test_y = data_loader.get_MNIST(n_dim, "AE_CNN", [
                                                                 1, 0, 6, 9])
        classical = classicalNC()
        classical.fit(train_X, train_y)
        labels_4.append(classical.predict(test_X))

        train_X, train_y, test_X, test_y = data_loader.get_MNIST(
            n_dim, "AE_CNN", [1, 0, 4, 5, 6, 9])
        classical = classicalNC()
        classical.fit(train_X, train_y)
        labels_6.append(classical.predict(test_X))


    variance_error_graph(test_y, "./clasico_vs_cuantico_MNIST.png",
                         ("2 clases", labels_2), ("4 clases", labels_4), ("6 clases", labels_6))"""