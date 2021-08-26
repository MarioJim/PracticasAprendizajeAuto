import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from graphs import graphConfusionMatrix

print(" ~ Reading default.txt and generating train and test sets")
data = pd.read_csv('default.txt', sep="	")
# Transform 'default' column from Yes/No to a boolean
data["default"] = (data["default"] == "Yes").astype(bool)
# Transform 'student' column from Yes/No to an integer
data["student"] = (data["student"] == "Yes").astype(int)
x = data.iloc[:, 1:4].values.reshape(-1, 3)
y = data.iloc[:, 0].values.reshape(-1, 1)
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)

print(" ~ Creating sklearn's logistic regression model")
regressor = LogisticRegression()
regressor.fit(xTrain, yTrain.ravel())

print(" ~ Testing sklearn's logistic regression model")
yPredicted = regressor.predict(xTest)
accuracy = accuracy_score(yTest, yPredicted)
print("   → Accuracy:", accuracy)
cm = confusion_matrix(yTest, yPredicted)
print("   → Confusion matrix:", cm.tolist())

graphConfusionMatrix(cm, ["Yes", "No"], "Default",
                     "DEFAULT w/sklearn", "default_cm_sklearn.png")
