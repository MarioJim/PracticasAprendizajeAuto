import sys
import matplotlib.pyplot as plt
import numpy as np

shouldDisplay = "--display-graphs" in sys.argv
shouldSave = "--save-graphs" in sys.argv


def graphGENERO(height, weight, theta, title, filename):
    fig, ax = plt.subplots()
    ax.set_title("GENERO: " + title)
    ax.set_xlabel("Height normalized")
    ax.set_ylabel("Weight")

    # Add scatter plot
    ax.scatter(height, weight, label="Datapoints")

    # Add line from theta parameters
    points = 60
    lineX = np.linspace(height.min(), height.max(), points)
    constant = np.ones(points)
    lineY = np.dot(np.vstack((lineX, constant)).T, theta)
    equation = "y = {:.2f}x + {:.2f}".format(theta[0], theta[1])
    ax.plot(lineX, lineY, 'g', label=equation)

    ax.legend()
    if shouldSave:
        fig.savefig(filename)
    if shouldDisplay:
        plt.show()


def graphMTCARS(disp, wt, hp, theta, title, filename):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title("MTCARS: " + title)
    ax.set_xlabel("Displacement normalized")
    ax.set_ylabel("Weight normalized")
    ax.set_zlabel("Horsepower")

    # Add scatter plot
    ax.scatter(disp, wt, hp, label="Datapoints")

    # Add line from theta parameters
    points = 60
    lineX = np.linspace(disp.min(), disp.max(), points)
    lineY = np.linspace(wt.min(), wt.max(), points)
    constant = np.ones(points)
    lineZ = np.dot(np.vstack((lineX, lineY, constant)).T, theta)
    equation = "z = {:.2f}x + {:.2f}y + {:.2f}".format(
        theta[0], theta[1], theta[2])
    ax.plot(lineX, lineY, lineZ, 'g', label=equation)

    ax.legend()
    if shouldSave:
        fig.savefig(filename)
    if shouldDisplay:
        plt.show()
