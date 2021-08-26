import sys
import matplotlib.pyplot as plt
import numpy as np

shouldDisplay = "--display-graphs" in sys.argv
shouldSave = "--save-graphs" in sys.argv


def graphDEFAULT(height, weight, theta, title, filename):
    # TODO
    pass


def graphGENERO(height, weight, predictions, title, filename):
    fig, ax = plt.subplots()
    ax.set_title("GENERO: " + title)
    ax.set_xlabel("Height")
    ax.set_ylabel("Weight")

    # TODO

    if shouldSave:
        fig.savefig(filename)
    if shouldDisplay:
        plt.show()
