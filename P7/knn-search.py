from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

digitsX, digitsy = load_digits(return_X_y=True)
trainX, testX, trainy, testy = train_test_split(
    digitsX, digitsy, test_size=0.2, random_state=0)

for neighbors in [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 50]:
    clf = KNeighborsClassifier(n_neighbors=neighbors)
    clf.fit(trainX, trainy)
    accuracy = clf.score(testX, testy)
    print("Accuracy using {:2} neighbors: {:.4f}".format(neighbors, accuracy))
