import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model

dataframe = pd.read_csv('FuelConsumptionCo2.csv')
cdf = dataframe[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY',
                 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

msk = np.random.rand(len(dataframe)) < 0.8
train = cdf[msk]
test = cdf[~msk]

regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x, y)

y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']].values)
test_x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

if __name__ == '__main__':
    # print("Residual sum of squares: %.2f" % np.mean((y_hat - y) ** 2))
    # print("Residual sum of squares: %.2f" % np.mean((y_hat - y) ** 2))
    print("Residual sum of squares: %.2f" % np.mean((y_hat - test_y) ** 2))
    print("Variance score: %.2f" % regr.score(test_x, test_y))
    print()
