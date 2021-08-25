import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from GradientDescent import GradientDescent
from graphs import graphGENERO

print(" ~ Reading genero.txt and generating train and test sets")
data = pd.read_csv('genero.txt')
height = data.iloc[:, 1].values.reshape(-1, 1)
weight = data.iloc[:, 2].values.reshape(-1, 1)
heightTrain, heightTest, weightTrain, weightTest = train_test_split(
    height, weight, test_size=0.2)

print(" ~ Creating gradient descent model")
regressor = GradientDescent()
regressor.fit(heightTrain, weightTrain.ravel())

print(" ~ Testing gradient descent model")
predictedWeight = regressor.predict(heightTest)
regressorMSE = mean_squared_error(weightTest, predictedWeight)
print("   → MSE:", regressorMSE)
regressorR2 = r2_score(weightTest, predictedWeight)
print("   → R2:", regressorR2)

normHeight = (height - regressor.mu) / regressor.sigma
graphGENERO(normHeight, weight, regressor.theta,
            "Gradient Descent", "genero_gradient.png")
