import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from GradientDescent import normalize
from graphs import graphGENERO

print(" ~ Reading genero.txt and generating train and test sets")
generoData = pd.read_csv('genero.txt')
height = generoData.iloc[:, 1].values.reshape(-1, 1)
height, _, _ = normalize(height)
weight = generoData.iloc[:, 2].values.reshape(-1, 1)
heightTrain, heightTest, weightTrain, weightTest = train_test_split(
    height, weight, test_size=0.2)

print(" ~ Creating linear regression model")
regressor = LinearRegression()
regressor.fit(heightTrain, weightTrain)

print(" ~ Testing linear regression model")
predictedWeight = regressor.predict(heightTest)
regressorMSE = mean_squared_error(weightTest, predictedWeight)
print("   → MSE:", regressorMSE)
regressorR2 = r2_score(weightTest, predictedWeight)
print("   → R2:", regressorR2)

theta = np.append(regressor.coef_.copy(), regressor.intercept_)
graphGENERO(height, weight, theta, "Linear Regression", "genero_linearReg.png")
