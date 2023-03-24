import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import sklearn.tree as tree

# specifies the character used to separate the values in the CSV file.
# In this case, it is set to ',' indicating that the values in the file are separated by commas.
# This is because CSV stands for "Comma-Separated Values",
# and comma is the most commonly used delimiter for CSV files.
# However, other delimiters such as tabs, semicolons, or spaces can also be used,
# and the delimiter parameter can be set accordingly.
my_data = pd.read_csv('../datasets/drug200.csv', delimiter=',')

# PREPROCESSING
X = my_data.drop(columns='Drug').values
le_sex = LabelEncoder()
le_sex.fit(['F', 'M'])
X[:, 1] = le_sex.transform(X[:, 1])

le_BP = LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:, 2] = le_BP.transform(X[:, 2])

le_Chol = LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:, 3] = le_Chol.transform(X[:, 3])

y = my_data['Drug'].values

# SETUP DECISION TREE
# NOTES... random_state is used to set the random seed used by the random number generator.
# This is important because when we split our data into training and testing sets,
# we want to ensure that the split is reproducible.
# This means that if we rerun our code with the same random_state value,
# we will get the same train/test split. This is useful for debugging, testing, and comparing different models.
# It is important to note that the random_state value does not directly affect the performance of the model.
# It only affects the way the data is split into training and testing sets.
# The performance of the model depends on a variety of factors, including the quality of the data,
# the choice of features, the complexity of the model, and the optimization of the model's hyperparameters.

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
drugTree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
drugTree.fit(X_trainset, y_trainset)
predTree = drugTree.predict(X_testset)

# EVALUATION
score = accuracy_score(y_testset, predTree) * 100
print("Decision tree's accuracy: %.2f" % score)

# VISUALIZATION
tree.plot_tree(drugTree)
plt.show()
if __name__ == '__main__':
    print()
