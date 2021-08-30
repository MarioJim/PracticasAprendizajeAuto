import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from GradientDescent import GradientDescent
from graphs import graphConfusionMatrix, graphGENERO

print(" ~ Reading default.txt and generating train and test sets")
data = pd.read_csv('default.txt', sep="	")
# Transform 'default' column from Yes/No to a boolean
data["default"] = (data["default"] == "Yes").astype(bool)
# Transform 'student' column from Yes/No to an integer
data["student"] = (data["student"] == "Yes").astype(int)
x = data.iloc[:, 1:4].values.reshape(-1, 3)
y = data.iloc[:, 0].values.reshape(-1, 1)
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)

# Convert Bool true/false to Int 1/0
yTrainToInt = (yTrain.ravel() == True).astype(int) # True = 1, False = 0
yTest = yTest.ravel()

print(" ~ Creating our logistric regression model with gradient descent")
regressor = GradientDescent()
regressor.fit(xTrain, yTrainToInt)

print(" ~ Testing our logistric regression model with gradient descent")
yPredicted = regressor.predict(xTest)

# Convert Int 1/0 to True / False
yPredicted = np.array(list(map((lambda x : True if x == 1 else False), yPredicted)))

accuracy = accuracy_score(yTest, yPredicted)
print("   → Accuracy:", accuracy)
cm = confusion_matrix(yTest, yPredicted)
print("   → Confusion matrix:", cm.tolist())
graphConfusionMatrix(cm, ["Yes", "No"], "Default",
                     "DEFAULT w/sklearn", "default_cm_sklearn.png")