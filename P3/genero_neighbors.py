import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from graphs import graphConfusionMatrix, graphNeighborsAccuracy, graphGENERO

print(" ~ Reading genero.txt and generating train and test sets")
data = pd.read_csv('genero.txt')
x = data.iloc[:, 1:3].values.reshape(-1, 2)
y = data.iloc[:, 0].values.reshape(-1, 1)
xTrain, xTest, yTrain, yTest = train_test_split(
    x, y, test_size=0.2, random_state=0)


yIntToStr = np.vectorize(lambda x: ["Female", "Male"][int(x)])

print(" ~ Creating and testing the k-NN models for different values")
print("   → neighbors   accuracy       confusion matrix")
k_neighbors_vals = [1, 2, 3, 5, 10, 15, 20, 50, 75, 100]
accuracy_vals = []
best_k_val = (-1, 0, [], [])  # k_neighbors, accuracy, conf_matrix, yPredicted
for val in k_neighbors_vals:
    nn = KNeighborsRegressor(n_neighbors=val, algorithm="brute")
    # Convert String Male/Female to integers 1/0
    yTrainInt = (yTrain.ravel() == "Male").astype(int)
    nn.fit(xTrain, yTrainInt)
    yPredicted = nn.predict(xTest)
    yPredicted = yIntToStr(yPredicted)
    accuracy = accuracy_score(yTest, yPredicted)
    accuracy_vals.append(accuracy)
    cm = confusion_matrix(yTest, yPredicted)
    print("   → {:^9}   {:^8}   {}".format(val, accuracy, cm.tolist()))
    if best_k_val[1] < accuracy:
        best_k_val = (val, accuracy, cm, yPredicted)

graphNeighborsAccuracy(k_neighbors_vals, accuracy_vals,
                       "GENERO", "genero_neighbors_acc.png")
graphGENERO(xTest, yTest, best_k_val[3], "{}-Nearest Neighbors".format(best_k_val[0]),
            "genero_neighbors_pred.png")
graphConfusionMatrix(best_k_val[2], ["Female", "Male"], "Gender",
                     "GENERO: {}-Nearest Neighbors".format(best_k_val[0]),
                     "genero_neighbors_cm.png")
