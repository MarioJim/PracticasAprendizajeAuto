import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

from GradientDescent import GradientDescent
from graphs import graphMTCARS

print(" ~ Reading mtcars.txt")
data = pd.read_csv('mtcars.txt', sep=' ')

print(" ~ Creating the k-fold cross validation model")
kfold = KFold()
mses = []
r2s = []

print(" ~ Creating the gradient descent model for every train/test division")
for trainIdxs, testIdxs in kfold.split(data):
    # Fit model to data
    train = data.iloc[trainIdxs]
    trainX = train[["disp", "wt"]]
    trainY = train[["hp"]]
    regressor = GradientDescent()
    regressor.fit(trainX, trainY.values.ravel())

    # Graph model
    idx = len(mses) + 1
    normX = (data[["disp", "wt"]] - regressor.mu) / regressor.sigma
    graphMTCARS(normX["disp"], normX["wt"], data["hp"], regressor.theta,
                "Gradient Descent, fold {}".format(idx),
                "mtcars_gradient_{}.png".format(idx))

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

print(" ~ Calculating final MSE")
print("   → Final MSE:", np.average(mses))
print("   → Final R2:", np.average(r2s))
