import numpy as np
import matplotlib.pyplot as plt

def error_graph(y_test: np.ndarray, *args):
    names = []
    accuracy = []
    for prediction in args:
        names.append(prediction[0])
        counts = 0
        for i, y in enumerate(prediction[1]):
            if y == y_test[i]:
                counts += 1
        accuracy.append(counts / len(y_test))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Method used')
    ax.set_title('Scores by group and gender')
    ax.bar(names,accuracy)
    plt.savefig('accuracy_methods.png')
    plt.show()

