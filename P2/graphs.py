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
    plt.tight_layout()
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
