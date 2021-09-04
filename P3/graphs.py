import math
import sys
import matplotlib.pyplot as plt
import numpy as np

shouldDisplay = "--display-graphs" in sys.argv
shouldSave = "--save-graphs" in sys.argv

mapColor = np.vectorize(lambda x: "b" if x == "Male" else "r")


def graphGENERO(xys, actualVals, predictedVals, title, filename):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.set_size_inches(6, 7)
    fig.suptitle("GENERO: " + title)

    ax1.set_title("Actual values")
    ax1.set_ylabel("Weight")
    actualColors = mapColor(actualVals).flatten()
    ax1.scatter(xys[:, 0], xys[:, 1], c=actualColors)

    ax2.set_title("Predicted values")
    ax2.set_xlabel("Height")
    ax2.set_ylabel("Weight")
    predictedColors = mapColor(predictedVals).flatten()
    ax2.scatter(xys[:, 0], xys[:, 1], c=predictedColors)

    if shouldSave:
        fig.savefig(filename)
    if shouldDisplay:
        plt.show()


def graphConfusionMatrix(cm, labels, variableName, title, filename):
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap="BuGn")
    plt.title("Confusion matrix: " + title)
    plt.colorbar()
    plt.xticks(np.arange(2), labels, size=10)
    plt.yticks(np.arange(2), labels, size=10)
    plt.xlabel("Actual " + variableName)
    plt.ylabel("Predicted " + variableName)

    for x in range(2):
        for y in range(2):
            plt.annotate(cm[x][y], xy=(y, x),
                         horizontalalignment="center",
                         verticalalignment="center")

    if shouldSave:
        plt.savefig(filename)
    if shouldDisplay:
        plt.show()


def graphNeighborsAccuracy(k_neighbors, accuracy, title, filename):
    x = list(map(str, k_neighbors))

    plt.figure()
    plt.title(title + ": k-Neighbors vs Accuracy")
    plt.plot(x, accuracy)
    plt.plot(x, accuracy, 'og')

    for i in range(len(x)):
        xy = (x[i], accuracy[i])
        plt.annotate("{:.3f}".format(
            accuracy[i]), xy, xycoords='data', xytext=(-12, -16), textcoords="offset points")

    plt.xlabel("k-Neighbors")
    plt.ylabel("Accuracy")
    bottom_ylim = math.floor(10 * min(accuracy)) / 10
    top_ylim = math.ceil(10 * max(accuracy)) / 10
    plt.ylim((bottom_ylim, top_ylim))

    if shouldSave:
        plt.savefig(filename)
    if shouldDisplay:
        plt.show()
