import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

# LINEAR MODEL REGRESSION EXAMPLE

dataframe = pd.read_csv('FuelConsumptionCo2.csv')
selected_features = dataframe[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# split dataset into train and test sets.80% of the entire dataset will be used for training and 20% for testing
# this generates a random array of the same length as the dataframe.
# the comparison with 0.8  generates a boolean array where True values indicate that
# the corresponding row should be included in the training set and false values
# indicates that the row should be in the test set
msk = np.random.rand(len(dataframe)) < 0.8
# selects the rows from selected_features where msk is True to create the training set
train = selected_features[msk]
# selects the rows where msk is False to create the test set.
test = selected_features[~msk]

# modelling
regr = linear_model.LinearRegression()
train_x = np.asarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

# plot the fit line over the data
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
fit_line = regr.intercept_[0] + (regr.coef_[0][0] * train_x)
plt.plot(train_x, fit_line, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

# Evaluation
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)
if __name__ == '__main__':
    # plt.show()
    # print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
    # print("R2-score: %.2f" % r2_score(test_y, test_y_))
    print(selected_features.head(n=9))
    # print(np.mean([0, 1, 2, 3]))
    # print("Coefficients: ", regr.coef_)
    # print("Intercept: ", regr.intercept_)
