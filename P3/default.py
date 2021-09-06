import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from graphs import graphConfusionMatrix, graphNeighborsAccuracy

print(" ~ Reading default.txt and generating train and test sets")
data = pd.read_csv('default.txt', sep="	")
# Transform 'default' and 'student' columns from Yes/No to integers (1/0)
data["default"] = (data["default"] == "Yes").astype(int)
data["student"] = (data["student"] == "Yes").astype(int)
x = data.iloc[:, 1:4].values.reshape(-1, 3)
y = data.iloc[:, 0].values.reshape(-1, 1)
xTrain, xTest, yTrain, yTest = train_test_split(
    x, y, test_size=0.2, random_state=0)


# Nearest neighbors
print(" ~ Creating and testing the k-NN models for different values")
print("   → neighbors   accuracy       confusion matrix")
k_neighbors_vals = [1, 2, 3, 5, 10, 15, 20, 50, 75, 100]
accuracy_vals = []
best_k_val = (-1, 0, [])  # k_neighbors, accuracy, conf_matrix
for val in k_neighbors_vals:
    nn = KNeighborsRegressor(n_neighbors=val, algorithm="brute")
    nn.fit(xTrain, yTrain)
    yPredicted = nn.predict(xTest).astype(int)
    accuracy = accuracy_score(yTest, yPredicted)
    accuracy_vals.append(accuracy)
    cm = confusion_matrix(yTest, yPredicted)
    print("   → {:^9}   {:^8}   {}".format(val, accuracy, cm.tolist()))
    if best_k_val[1] < accuracy:
        best_k_val = (val, accuracy, cm)

graphNeighborsAccuracy(k_neighbors_vals, accuracy_vals,
                       "DEFAULT", "default_neighbors_acc.png")
graphConfusionMatrix(best_k_val[2], ["Yes", "No"], "Default",
                     "DEFAULT: {}-Nearest Neighbors".format(best_k_val[0]),
                     "default_neighbors_cm.png")


# Logistic regression
print("\n\n ~ Creating the logistic regression model")
regressor = LogisticRegression()
regressor.fit(xTrain, yTrain.ravel())
print(" ~ Testing the logistic regression model")
yPredicted = regressor.predict(xTest)
accuracy = accuracy_score(yTest, yPredicted)
print("   → Accuracy:", accuracy)
cm = confusion_matrix(yTest, yPredicted)
print("   → Confusion matrix:", cm.tolist())

graphConfusionMatrix(cm, ["Yes", "No"], "Default",
                     "DEFAULT: Logistic Regression", "default_regression_cm.png")
