import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from GradientDescent import GradientDescent
from graphs import graphConfusionMatrix, graphGENERO

print(" ~ Reading genero.txt and generating train and test sets")
data = pd.read_csv('genero.txt')
x = data.iloc[:, 1:3].values.reshape(-1, 2)
y = data.iloc[:, 0].values.reshape(-1, 1)
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)

print(" ~ Creating our logistric regression model with gradient descent")
regressor = GradientDescent()
regressor.fit(xTrain, yTrain.ravel())

print(" ~ Testing our logistric regression model with gradient descent")
yPredicted = regressor.predict(xTest)
accuracy = accuracy_score(yTest, yPredicted)
print("   → Accuracy:", accuracy)
cm = confusion_matrix(yTest, yPredicted)
print("   → Confusion matrix:", cm.tolist())

graphGENERO(xTest, yTest, yPredicted, "Logistic Regression using gradient descent",
            "genero_sklearn.png")
graphConfusionMatrix(cm, ["Yes", "No"], "Gender",
                     "GENERO with gradient descent", "genero_cm_gradient.png")
