import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

from GradientDescent import normalize
from graphs import graphMTCARS

print(" ~ Reading mtcars.txt")
data = pd.read_csv('mtcars.txt', sep=' ')
data["disp"], _, _ = normalize(data["disp"])
data["wt"], _, _ = normalize(data["wt"])

print(" ~ Creating the k-fold cross validation model")
kfold = KFold()
mses = []
r2s = []

print(" ~ Creating the regression model for every train/test division")
for trainIdxs, testIdxs in kfold.split(data):
    # Fit model to data
    train = data.iloc[trainIdxs]
    trainX = train[["disp", "wt"]]
    trainY = train[["hp"]]
    regressor = LinearRegression()
    regressor.fit(trainX, trainY)

    # Graph model
    idx = len(mses) + 1
    theta = np.append(regressor.coef_.copy(), regressor.intercept_)
    graphMTCARS(trainX["disp"], trainX["wt"], trainY["hp"],
                theta, "Linear Regression, fold {}".format(idx),
                "mtcars_linearReg_{}.png".format(idx))

    # Predict data using model
    test = data.iloc[testIdxs]
    testX = test[["disp", "wt"]]
    testY = test[["hp"]]
    predictedY = regressor.predict(testX)

    # Score predicted data
    regressorMSE = mean_squared_error(testY, predictedY)
    regressorR2 = r2_score(testY, predictedY)
    print("   → Partial MSE:", regressorMSE, "\tPartial R2: ", regressorR2)
    mses.append(regressorMSE)
    r2s.append(regressorR2)

print(" ~ Calculating final scores")
print("   → Final MSE:", np.average(mses))
print("   → Final R2:", np.average(r2s))
