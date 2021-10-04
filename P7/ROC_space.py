import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle

def calculateROC(testy, predicty):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve(testy[:, i], predicty[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return [classifier_name, fpr["macro"], tpr["macro"], roc_auc["macro"]]


# Plot all ROC curves
def plotROCs(ROCs):
    lw = 2
    plt.figure(figsize=(12, 7), dpi=100)

    colors = cycle(['black', 'darkorange', 'cornflowerblue', 'olive', 'gray', 'rosybrown', 'orange', 'darkviolet', 'crimson', 'slategray'])
    for i, color in zip(range(len(ROCs)), colors):
        plt.plot(ROCs[i][1], ROCs[i][2], color=color, lw=lw,
            label=ROCs[i][0] + ' (area = {0:0.2f})'
                    ''.format(ROCs[i][3]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Espacio ROC comparando los diferentes clasificadores')
    plt.legend(loc="lower right")
    plt.savefig("ROC_curve.png")
    plt.show()

ROCs = []
n_classes = 10
clfs = ["linear", "poly", "rbf", "sigmoid",
                 "logistic", "knn", "bayes"]

# Load the dataset and split it
digitsX, digitsy = load_digits(return_X_y=True)
digitsy = label_binarize(digitsy, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
trainX, testX, trainy, testy = train_test_split(
    digitsX, digitsy, test_size=0.2, random_state=0)

for j in range(len(clfs)):
    if clfs[j] in ["linear", "poly", "rbf", "sigmoid"]:
        clf = SVC(kernel=clfs[j])
        # Fit the model and predict the labels
        predicty = OneVsRestClassifier(clf).fit(trainX, trainy).decision_function(testX)
        classifier_name = "SVM classifier with a " + {
            "linear": "linear kernel",
            "poly": "polynomial kernel",
            "rbf": "Radial Basis Function kernel",
            "sigmoid": "sigmoid kernel",
        }[clfs[j]]
    elif clfs[j] == "logistic":
        clf = LogisticRegression(multi_class="ovr", max_iter=1000)
        # Fit the model and predict the labels
        predicty = OneVsRestClassifier(clf).fit(trainX, trainy).decision_function(testX)
        classifier_name = "Logistic regression classifier"
    elif clfs[j] == "knn":
        clf = KNeighborsClassifier(n_neighbors=1)
        # Fit the model and predict the labels
        predicty = OneVsRestClassifier(clf).fit(trainX, trainy).predict(testX)
        classifier_name = "k-Nearest Neighbors classifier"
    elif clfs[j] == "bayes":
        clf = BernoulliNB()
        # Fit the model and predict the labels
        predicty = OneVsRestClassifier(clf).fit(trainX, trainy).predict(testX)
        classifier_name = "Naive Bayes classifier"
    
    ROCs.append(calculateROC(testy, predicty))

plotROCs(ROCs)