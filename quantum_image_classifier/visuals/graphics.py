import numpy as np
import matplotlib.pyplot as plt

def error_graph(y_test: np.ndarray, img_name: str, title:str, *args):
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
    ax.set_title(title)
    ax.bar(names,accuracy)
    plt.savefig(img_name)

def variance_error_graph(y_test: np.ndarray, img_name: str, *args):
    names = []
    accuracy = []
    CTEs = []
    error = []
    for prediction in args:
        names.append(prediction[0])
        for iteration in prediction[1]:
            counts = 0
            for i, y in enumerate(iteration):
                if y == y_test[i]:
                    counts += 1
            accuracy.append(counts / len(y_test))
        CTEs.append(np.mean(accuracy))
        error.append(np.std(accuracy))

    x_pos = np.arange(len(args))

    # Build the plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    for i,j in zip(x_pos, CTEs):
        ax.annotate(str(j),xy=(i,j))
    ax.set_ylabel('Accuracy')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig(img_name)
    plt.show()
    
