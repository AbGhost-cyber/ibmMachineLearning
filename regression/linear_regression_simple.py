import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataframe = pd.read_csv("../datasets/FuelConsumptionCo2.csv")
dataframe = dataframe[['ENGINESIZE', 'CO2EMISSIONS']]

X = np.asanyarray(dataframe['ENGINESIZE'])
X = X.reshape(-1, 1)
y = np.asanyarray(dataframe['CO2EMISSIONS'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

regr = LinearRegression()
regr.fit(X_train, y_train)

yhat = regr.predict(X_test)

if __name__ == '__main__':
    print(r2_score(y_test, yhat))
