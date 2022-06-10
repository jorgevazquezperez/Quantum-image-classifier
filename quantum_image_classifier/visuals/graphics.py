import numpy as np
import matplotlib.pyplot as plt
import random

def error_graph(img_name: str, *args: tuple) -> None:
    """
    Function to show the accuracy of one iteration.

    Args:
        y_test: labels of the test set
        *args: tuples with the names of the method (reduction method, number of clusters...) 
        and its prediction
    """

    # Calculate the accuracy
    names = []
    accuracy = []
    for prediction in args:
        names.append(prediction[0])
        accuracy.append(prediction[1])

    # Show and save the plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Method used')
    ax.set_title('Scores by group and gender')
    ax.bar(names,accuracy)
    plt.savefig(img_name)
    plt.show()
    plt.close(fig)


def variance_error_graph(img_name: str, *args: tuple) -> None:
    """
    Function to show the average accuracy of the method along with the standard deviation
    of multiple executions.

    Args:
        img_name: name to store the image
        *args: tuples with the names of the method (reduction method, number of clusters...) 
        and its accuracy precalculated
    """
    names = []
    CTEs = []
    error = []
    for prediction in args:
        names.append(prediction[0])
        CTEs.append(np.mean(prediction[1]))
        error.append(np.std(prediction[1]))

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
    plt.close(fig)

def cloud_point(data: np.ndarray, labels: np.ndarray, img_name: str, n_labels: int):
    """
    Function to show a cloud point plot with two clases with diferent colors.

    Args:
        data: data to be represented
        labels: labels associated to each point of data
        img_name: name to store the image
        n_labels: number of types of labels
    """
    colors = []
    type_color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for _ in range(n_labels)]
    for label in labels:
        colors.append(type_color[label])
    
    x = [i[0] for i in data]
    y = [i[1] for i in data]

    plt.scatter(x, y, c=colors)
    plt.show()
    plt.savefig(img_name)

