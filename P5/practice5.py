import os
import sys
from graphviz import Source
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def plot_decision_boundary(clf, X, y, axes=[0, 7, 0, 3]):
    x1s = np.linspace(axes[0], axes[1], 200)
    x2s = np.linspace(axes[2], axes[3], 200)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", label="Iris setosa")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", label="Iris versicolor")
    plt.plot(X[:, 0][y == 2], X[:, 1][y == 2], "g^", label="Iris virginica")
    plt.axis(axes)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="lower right", fontsize=14)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Setup folders
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

# Dataset selection
if len(sys.argv) < 2:
    print("Include an argument from the list [iris, wine, breast_cancer]")
    exit(1)

if sys.argv[1] == 'iris':
    iris = load_iris()
    feauture_names = iris.feature_names[-2:]
    target_names = iris.target_names
    treeFilename = 'iris_tree'
    X = iris.data[:, -2:]
    y = iris.target
elif sys.argv[1] == 'wine':
    wine = load_wine()
    feauture_names = wine.feature_names
    target_names = wine.target_names
    treeFilename = 'wine_tree'
    X = wine.data
    y = wine.target
elif sys.argv[1] == 'breast_cancer':
    breast_cancer = load_breast_cancer()
    feauture_names = breast_cancer.feature_names
    target_names = breast_cancer.target_names
    treeFilename = 'breast_cancer_tree'
    X = breast_cancer.data
    y = breast_cancer.target

xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=0)

# Create and train the decision tree model
tree_clf = DecisionTreeClassifier(random_state=0, max_leaf_nodes=6)
tree_clf.fit(xTrain, yTrain)

# Export an image of the tree
dot_src = export_graphviz(tree_clf, feature_names=feauture_names,
                          class_names=target_names, rounded=True,
                          filled=True)
image_filename = os.path.join(IMAGES_PATH, treeFilename)
Source(dot_src).render(image_filename, format="png", cleanup=True)

# Test the decision tree
accuracy = tree_clf.score(xTest, yTest)
print("Accuracy:", accuracy)


if sys.argv[1] == 'iris':
    plt.figure(figsize=(8, 4))
    plot_decision_boundary(tree_clf, xTest, yTest)
    plt.plot([0, 7], [0.8, 0.8], "k-", linewidth=2)
    plt.text(2.0, 1.0, "Depth=0", fontsize=15)
    plt.plot([4.95, 4.95], [0.8, 3], "k--", linewidth=2)
    plt.text(5.1, 1.9, "Depth=1", fontsize=13)
    save_fig("decision_tree_decision_boundaries_plot")
