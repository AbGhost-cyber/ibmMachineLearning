import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('../datasets/real_estate_data.csv')

# drop invalid rows(those with missing values) since we have sufficient data already
df.dropna(inplace=True)

X = df.drop(columns=['MEDV'])
Y = df['MEDV']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=1)

regression_tree = DecisionTreeRegressor(criterion="squared_error")

regression_tree.fit(X_train, Y_train)

r2_score = regression_tree.score(X_test, Y_test)

prediction = regression_tree.predict(X_test)

average_score = (prediction - Y_test).abs().mean()

print("$", average_score * 1000)
print("score", r2_score)

if __name__ == '__main__':
    print()
