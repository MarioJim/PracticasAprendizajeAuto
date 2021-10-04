import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Choose a classifier using the first parameter
possible_clfs = ["linear", "poly", "rbf", "sigmoid",
                 "logistic", "knn", "bayes"]
if len(sys.argv) < 2 or sys.argv[1] not in possible_clfs:
    print("Include an argument from the list:", possible_clfs)
    exit(1)

classifier_type = sys.argv[1]
if classifier_type in ["linear", "poly", "rbf", "sigmoid"]:
    clf = SVC(kernel=classifier_type)
    classifier_name = "SVM classifier with a " + {
        "linear": "linear kernel",
        "poly": "polynomial kernel",
        "rbf": "Radial Basis Function kernel",
        "sigmoid": "sigmoid kernel",
    }[classifier_type]
elif classifier_type == "logistic":
    clf = LogisticRegression(multi_class="ovr", max_iter=1000)
    classifier_name = "Logistic regression classifier"
elif classifier_type == "knn":
    clf = KNeighborsClassifier(n_neighbors=1)
    classifier_name = "k-Nearest Neighbors classifier"
elif classifier_type == "bayes":
    clf = BernoulliNB()
    classifier_name = "Naive Bayes classifier"

# Load the dataset and split it
digitsX, digitsy = load_digits(return_X_y=True)
trainX, testX, trainy, testy = train_test_split(
    digitsX, digitsy, test_size=0.2, random_state=0)

# Fit the model and predict the labels
clf.fit(trainX, trainy)
predicty = clf.predict(testX)

# Print results
print(classifier_name)
print("Accuracy:", accuracy_score(testy, predicty))
cm = confusion_matrix(testy, predicty)
print("Confusion matrix:")
print(cm)

# Plot the confusion matrix
plt.figure()
plt.imshow(cm, interpolation="nearest", cmap="BuGn")
plt.title("Confusion matrix using " + classifier_name)
plt.colorbar()
plt.xticks(np.arange(10), np.arange(10).astype(str), size=10)
plt.yticks(np.arange(10), np.arange(10).astype(str), size=10)
plt.xlabel("Actual digit")
plt.ylabel("Predicted digit")
for x in range(10):
    for y in range(10):
        plt.annotate(cm[x][y], xy=(y, x),
                     horizontalalignment="center",
                     verticalalignment="center")
plt.savefig(classifier_type + "_cm.png")
